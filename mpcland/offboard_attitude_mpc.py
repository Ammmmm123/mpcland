#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from px4_msgs.msg import OffboardControlMode, TrajectorySetpoint, VehicleCommand, SensorCombined,VehicleLocalPosition, VehicleStatus,VehicleRatesSetpoint
from envs import QuadrotorLandingEnv
from utils import QuadMPC
import numpy as np
import config.config as Config
import run_simulation_world_frame as run_simulation

#定义NED到ENU的转换函数
def ned_to_enu(ned_coords):
    """
    将NED坐标系(North-East-Down)转换为ENU坐标系(East-North-Up)
    
    参数:
    ned_coords: 包含NED坐标的元组、列表或numpy数组 (x_ned, y_ned, z_ned)
                x_ned - 北向坐标
                y_ned - 东向坐标
                z_ned - 地向坐标（向下为正）
    
    返回:
    enu_coords: 包含ENU坐标的numpy数组 (x_enu, y_enu, z_enu)
                x_enu - 东向坐标
                y_enu - 北向坐标
                z_enu - 天向坐标（向上为正）
    """
    # 确保输入是numpy数组
    ned = np.array(ned_coords)
    
    # 执行坐标系转换
    # NED到ENU的转换矩阵:
    # [0, 1,  0]
    # [1, 0,  0]
    # [0, 0, -1]
    enu = np.array([
        ned[1],  # 东 = 原来的东
        ned[0],  # 北 = 原来的北
        -ned[2]  # 上 = -原来的下
    ])
    
    return enu

#定义FRD到FLU的角速度转换函数
def frd_to_flu_angular_rates(frd_rates):
    """
    将FRD（前-右-下）机体坐标系的角速度转换为FLU（前-左-上）机体坐标系的角速度
    
    参数:
    frd_rates: 包含FRD角速度的元组、列表或numpy数组 (p_frd, q_frd, r_frd)
               p_frd - 绕FRD X轴（前向）的滚转角速度
               q_frd - 绕FRD Y轴（右向）的俯仰角速度
               r_frd - 绕FRD Z轴（下向）的偏航角速度
    
    返回:
    flu_rates: 包含FLU角速度的numpy数组 (p_flu, q_flu, r_flu)
               p_flu - 绕FLU X轴（前向）的滚转角速度
               q_flu - 绕FLU Y轴（左向）的俯仰角速度
               r_flu - 绕FLU Z轴（上向）的偏航角速度
    """
    # 确保输入是numpy数组
    frd = np.array(frd_rates)
    
    # FRD到FLU的角速度转换
    # X轴（滚转）方向相同，保持不变
    # Y轴（俯仰）方向相反，取负
    # Z轴（偏航）方向相反，取负
    flu = np.array([
        frd[0],   # p_flu = p_frd (滚转角速度不变)
        -frd[1],  # q_flu = -q_frd (俯仰角速度反向)
        -frd[2]   # r_flu = -r_frd (偏航角速度反向)
    ])
    
    return flu
    
class MPC_OffboardControl(Node):
    """Node for controlling a vehicle in offboard mode."""

    def __init__(self,env: QuadrotorLandingEnv, mpc_solver: QuadMPC, simulation_params: dict) -> None:
        super().__init__('MPC_OffboardControl')

        #传入环境、MPC求解器和仿真参数
        self.env = env
        self.mpc_solver = mpc_solver
        self.simulation_params = simulation_params

        # Configure QoS profile for publishing and subscribing
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        # Create publishers
        self.offboard_control_mode_publisher = self.create_publisher(
            OffboardControlMode, '/fmu/in/offboard_control_mode', qos_profile)
        self.trajectory_setpoint_publisher = self.create_publisher(
            TrajectorySetpoint, '/fmu/in/trajectory_setpoint', qos_profile)
        self.vehicle_command_publisher = self.create_publisher(
            VehicleCommand, '/fmu/in/vehicle_command', qos_profile)
        self.vehicle_rates_setpoint_publisher = self.create_publisher(
            VehicleRatesSetpoint, '/fmu/in/vehicle_rates_setpoint', qos_profile)

        # Create subscribers
        self.vehicle_local_position_subscriber = self.create_subscription(
            VehicleLocalPosition, '/fmu/out/vehicle_local_position_v1', self.vehicle_local_position_callback, qos_profile)
        self.vehicle_status_subscriber = self.create_subscription(
            VehicleStatus, '/fmu/out/vehicle_status_v1', self.vehicle_status_callback, qos_profile)
        self.sensor_combined_subscriber = self.create_subscription(
            SensorCombined, '/fmu/out/sensor_combined', self.sensor_combined_callback, qos_profile)

        # Initialize variables
        self.offboard_setpoint_counter = 0
        self.vehicle_local_position = VehicleLocalPosition()
        self.vehicle_status = VehicleStatus()
        self.sensor_combined = SensorCombined()
        self.takeoff_height = -5.0
        self.mode=False  #True:角速度-推力控制模式 False:位置控制模式

        # Create a timer to publish control commands
        self.timer = self.create_timer(0.08, self.timer_callback)
    
    def vehicle_local_position_callback(self, vehicle_local_position):
        """Callback function for vehicle_local_position topic subscriber."""
        self.vehicle_local_position = vehicle_local_position

    def vehicle_status_callback(self, vehicle_status):
        """Callback function for vehicle_status topic subscriber."""
        self.vehicle_status = vehicle_status
    
    def sensor_combined_callback(self, sensor_combined):
        """Callback function for sensor_combined topic subscriber."""
        self.sensor_combined = sensor_combined    

    def arm(self):
        """Send an arm command to the vehicle."""
        self.publish_vehicle_command(
            VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM, param1=1.0)
        self.get_logger().info('Arm command sent')

    def disarm(self):
        """Send a disarm command to the vehicle."""
        self.publish_vehicle_command(
            VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM, param1=0.0)
        self.get_logger().info('Disarm command sent')

    def engage_offboard_mode(self):
        """Switch to offboard mode."""
        self.publish_vehicle_command(
            VehicleCommand.VEHICLE_CMD_DO_SET_MODE, param1=1.0, param2=6.0)
        self.get_logger().info("Switching to offboard mode")

    def land(self):
        """Switch to land mode."""
        self.publish_vehicle_command(VehicleCommand.VEHICLE_CMD_NAV_LAND)
        self.get_logger().info("Switching to land mode")

    def publish_offboard_control_heartbeat_signal(self,mode:bool=False):
        """Publish the offboard control mode."""
        if mode:
            msg = OffboardControlMode()
            msg.position = False
            msg.velocity = False
            msg.acceleration = False
            msg.attitude = False
            msg.body_rate = True
            msg.thrust_and_torque = True
            msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
            self.offboard_control_mode_publisher.publish(msg)
            self.get_logger().info('角速度-推力控制模式...')

        else:
            msg = OffboardControlMode()
            msg.position = True
            msg.velocity = False
            msg.acceleration = False
            msg.attitude = False
            msg.body_rate = False
            msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
            self.offboard_control_mode_publisher.publish(msg)
            self.get_logger().info('位置控制模式...')

    def publish_position_setpoint(self, x: float, y: float, z: float):
        """Publish the trajectory setpoint."""
        msg = TrajectorySetpoint()
        msg.position = [x, y, z]
        msg.yaw = 1.57079  # (90 degree)
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        self.trajectory_setpoint_publisher.publish(msg)
        self.get_logger().info(f"发送位置控制指令： {[x, y, z]}")

    def publish_rates_setpoint(self, roll: float, pitch: float, yaw: float):
        """Publish the rates setpoint."""
        msg = VehicleRatesSetpoint()
        msg.roll = roll
        msg.pitch = pitch
        msg.yaw = yaw
        msg.thrust_body =[0.0, 0.0, -0.729]  # Set a constant thrust value
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        msg.reset_integral = False
        self.vehicle_rates_setpoint_publisher.publish(msg)
        self.get_logger().info('发送角速度-推力控制指令...')

    def publish_vehicle_command(self, command, **params) -> None:
        """Publish a vehicle command."""
        msg = VehicleCommand()
        msg.command = command
        msg.param1 = params.get("param1", 0.0)
        msg.param2 = params.get("param2", 0.0)
        msg.param3 = params.get("param3", 0.0)
        msg.param4 = params.get("param4", 0.0)
        msg.param5 = params.get("param5", 0.0)
        msg.param6 = params.get("param6", 0.0)
        msg.param7 = params.get("param7", 0.0)
        msg.target_system = 1
        msg.target_component = 1
        msg.source_system = 1
        msg.source_component = 1
        msg.from_external = True
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        self.vehicle_command_publisher.publish(msg)

    def timer_callback(self) -> None:
        """Callback function for the timer."""
        self.publish_offboard_control_heartbeat_signal(self.mode)
        
        if self.offboard_setpoint_counter == 10:
            self.engage_offboard_mode()
            self.arm()

        if self.vehicle_local_position.z > self.takeoff_height and self.vehicle_status.nav_state == VehicleStatus.NAVIGATION_STATE_OFFBOARD:
            self.publish_position_setpoint(0.0, 0.0, self.takeoff_height)
            
        elif self.vehicle_local_position.z <= self.takeoff_height:
            # self.land()
            # # exit(0)
            self.mode=True  #切换到角速度-推力控制模式
            self.publish_rates_setpoint(0.0, 0.0, 0.5)


        if self.offboard_setpoint_counter < 11:
            self.offboard_setpoint_counter += 1


def main(args=None) -> None:

    # --- 1. 模拟参数 ---
    simulation_params = {
        'quad_init_position': np.array([0.0, 0.0, 5.0]),
        'quad_init_velocity': np.array([0.0, 0.0, 0.0]),
        'quad_init_quaternions': run_simulation.euler_to_quaternion(0, 0, np.deg2rad(0)),
        
        'platform_init_state': np.array([0.0, 0.0, 0.8, np.deg2rad(30)]),
        'platform_u1': 0.2,
        'platform_u2': np.deg2rad(-30.0)
    }

    # --- 2. 初始化环境和MPC控制器 ---
    env = QuadrotorLandingEnv(dt=Config.DELTA_T)
    mpc_solver = QuadMPC(horizon=Config.MPC.HORIZON, dt=Config.DELTA_T)

    # --- 3. 创建ROS节点 ---
    print('Starting offboard control node...')
    rclpy.init(args=args)
    offboard_control = MPC_OffboardControl(env,mpc_solver,simulation_params)
    rclpy.spin(offboard_control)
    offboard_control.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(e)
