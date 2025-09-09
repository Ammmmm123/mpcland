#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from px4_msgs.msg import OffboardControlMode, TrajectorySetpoint, VehicleCommand, SensorCombined,VehicleLocalPosition, VehicleStatus,VehicleRatesSetpoint,VehicleAttitude

from envs import QuadrotorLandingEnv
from utils import QuadMPC
import math
import numpy as np
import config.config as Config
import run_simulation_world_frame as run_simulation


# ==============================================================================
# 辅助函数 - 坐标系转换和归一化处理
# ==============================================================================



def quat_mul(q1, q2):
    """Hamilton convention quaternion multiplication: q = q1 ⊗ q2, both [w,x,y,z]."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return [
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ]

def quat_norm(q):
    w, x, y, z = q
    n = math.sqrt(w*w + x*x + y*y + z*z)
    if n == 0:
        raise ValueError("Zero-norm quaternion.")
    return [w/n, x/n, y/n, z/n]

def quat_from_axis_angle(axis, angle_rad):
    """axis: iterable of length 3, angle in radians. Returns [w,x,y,z]."""
    ax, ay, az = axis
    half = 0.5 * angle_rad
    s = math.sin(half)
    return quat_norm([math.cos(half), ax*s, ay*s, az*s])

# 预定义旋转四元数
Q_X_180 = quat_from_axis_angle((1.0, 0.0, 0.0), math.pi)          # R_x(pi)
Q_Z_90  = quat_from_axis_angle((0.0, 0.0, 1.0), math.pi * 0.5)    # R_z(pi/2)

def frd_ned_to_flu_enu(q_frd_ned):
    """
    Convert a quaternion from FRD->NED convention to FLU->ENU convention.
    Input/Output quaternions are [w, x, y, z], Hamilton, active rotations.
    """
    q = quat_norm(q_frd_ned)
    # q_target = q_z90 ⊗ q_x180 ⊗ q ⊗ q_x180
    q_out = quat_mul(Q_Z_90, quat_mul(Q_X_180, quat_mul(q, Q_X_180)))
    return quat_norm(q_out)


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
    
#归一化FLU角速度 TO FRD角速度
def flu_normalized_to_frd_omega(flu_normalized):
    """
    将FLU（前-左-上）机体坐标系的归一化角速度转换为FRD（前-右-下）机体坐标系的角速度
    
    参数:
    flu_normalized: 包含FLU归一化角速度的元组、列表或numpy数组 (p_flu_norm, q_flu_norm, r_flu_norm)
                    p_flu_norm - 绕FLU X轴（前向）的归一化滚转角速度 [-1, 1]
                    q_flu_norm - 绕FLU Y轴（左向）的归一化俯仰角速度 [-1, 1]
                    r_flu_norm - 绕FLU Z轴（上向）的归一化偏航角速度 [-1, 1]
    
    返回:
    frd_omega: 包含FRD角速度的numpy数组 (p_frd, q_frd, r_frd)
               p_frd - 绕FRD X轴（前向）的滚转角速度 (rad/s)
               q_frd - 绕FRD Y轴（右向）的俯仰角速度 (rad/s)
               r_frd - 绕FRD Z轴（下向）的偏航角速度 (rad/s)
    """
    # 确保输入是numpy数组
    flu_norm = np.array(flu_normalized)
    
    # 确保输入值在[-1, 1]范围内
    flu_norm = np.clip(flu_norm, -1.0, 1.0)
    
    # FLU到FRD的角速度转换
    # X轴（滚转）方向相同，保持不变
    # Y轴（俯仰）方向相反，取负
    # Z轴（偏航）方向相反，取负
    frd_norm = np.array([
        flu_norm[0],   # p_frd_norm = p_flu_norm (滚转角速度不变)
        -flu_norm[1],  # q_frd_norm = -q_flu_norm (俯仰角速度反向)
        -flu_norm[2]   # r_frd_norm = -r_flu_norm (偏航角速度反向)
    ])
    
    # 将归一化值转换为实际角速度
    frd_omega = frd_norm * Config.Quadrotor.OMEGA_MAX
    
    return frd_omega

# ==============================================================================
# MPC Offboard Control Node 建立
# ==============================================================================

class MPC_OffboardControl(Node):
    """Node for controlling a vehicle in offboard mode."""

    def __init__(self,env: QuadrotorLandingEnv, mpc_solver: QuadMPC, simulation_params: dict) -> None:
        super().__init__('MPC_OffboardControl')

    # ==============================================================================
    # MPC参数传入
    # ==============================================================================

        #传入环境、MPC求解器和仿真参数
        self.env = env
        self.mpc_solver = mpc_solver
        self.simulation_params = simulation_params

        #初始化环境和历史记录
        reset_params = {k: v for k, v in self.simulation_params.items() if 'quad' in k or 'platform_init' in k}
        self.platform_control = np.array([self.simulation_params['platform_u1'], self.simulation_params['platform_u2']])  
        self.obs, self.info = self.env.reset(**reset_params)
        self.history = {
            'time': [], 'quad_pos': [], 'quad_vel': [], 'quad_quat': [],
            'plat_pos': [], 'plat_vel': [], 'plat_psi': [],
            'rel_pos': [], 'control_input': []
        }
        
        # 预加载MPC代价函数矩阵
        self.nx, self.nu, self.N = self.mpc_solver.nx, self.mpc_solver.nu, self.mpc_solver.N
        self.q_weights = np.array(Config.MPC.STATE_WEIGHTS)   # 状态权重向量
        self.r_weights = np.array(Config.MPC.CONTROL_WEIGHTS) # 控制权重向量
        self.step=0

    # ==============================================================================
    # px4通信接口建立
    # ==============================================================================

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
        self.vehicle_attitude_subscriber = self.create_subscription(
            VehicleAttitude, '/fmu/out/vehicle_attitude', self.vehicle_attitude_callback, qos_profile)

        # Initialize variables
        self.offboard_setpoint_counter = 0
        self.vehicle_local_position = VehicleLocalPosition()
        self.vehicle_status = VehicleStatus()
        self.sensor_combined = SensorCombined()
        self.vehicle_attitude = VehicleAttitude()
        self.takeoff_height = -5.0
        self.mode=False  #True:角速度-推力控制模式 False:位置控制模式

        # Create a timer to publish control commands
        self.timer = self.create_timer(Config.DELTA_T, self.timer_callback)
    
    def vehicle_local_position_callback(self, vehicle_local_position):
        """Callback function for vehicle_local_position topic subscriber."""
        self.vehicle_local_position = vehicle_local_position

    def vehicle_status_callback(self, vehicle_status):
        """Callback function for vehicle_status topic subscriber."""
        self.vehicle_status = vehicle_status
    
    def sensor_combined_callback(self, sensor_combined):
        """Callback function for sensor_combined topic subscriber."""
        self.sensor_combined = sensor_combined    

    def vehicle_attitude_callback(self, vehicle_attitude):
        """Callback function for vehicle_attitude topic subscriber."""
        self.vehicle_attitude = vehicle_attitude

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

    def publish_rates_setpoint(self,thhrust_z:float, roll: float, pitch: float, yaw: float):
        """Publish the rates setpoint."""
        thhrust_z = np.clip(thhrust_z, -1.0, 1.0)
        msg = VehicleRatesSetpoint()
        msg.roll = roll
        msg.pitch = pitch
        msg.yaw = yaw
        msg.thrust_body =[0.0, 0.0, -thhrust_z]  # Set a constant thrust value起飞值-0.7289
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        msg.reset_integral = False
        self.vehicle_rates_setpoint_publisher.publish(msg)
        self.get_logger().info('发送角速度-推力控制指令...')
        self.get_logger().info(str(self.vehicle_local_position.z))

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

    # ==============================================================================
    # 回调函数--定时执行actions
    # ==============================================================================

    def timer_callback(self) -> None:
        """Callback function for the timer."""

        #发送心跳信号，保持offboard模式
        self.publish_offboard_control_heartbeat_signal(self.mode)

        #发送10次心跳信号后，切换到offboard模式并解锁
        if self.offboard_setpoint_counter == 10:
            self.engage_offboard_mode()
            self.arm()

        if self.vehicle_local_position.z > self.takeoff_height and self.vehicle_status.nav_state == VehicleStatus.NAVIGATION_STATE_OFFBOARD and self.mode==False:
            self.publish_position_setpoint(0.0, 0.0, self.takeoff_height)

            
        elif self.vehicle_local_position.z <= self.takeoff_height or self.mode==True:
            self.mode=True  #切换到角速度-推力控制模式
            self.get_logger().info('进入跟踪降落模式...')
            # 步骤 3.1: 获取当前世界状态            
            quad_world_state = np.concatenate([
                ned_to_enu([self.vehicle_local_position.x,
                self.vehicle_local_position.y,
                self.vehicle_local_position.z]),
                ned_to_enu([self.vehicle_local_position.vx,
                self.vehicle_local_position.vy,
                self.vehicle_local_position.vz]),
                frd_ned_to_flu_enu(self.vehicle_attitude.q)]
            )
            current_platform_state = self.env.platform.state

            print(f"当前无人机位置: {quad_world_state[:3]}")
            # 步骤 3.2: 预测平台未来轨迹，并生成MPC参考轨迹 (调用核心算法)
            platform_traj_pred = run_simulation.predict_platform_trajectory_world(
                current_platform_state, self.platform_control, self.N, self.env.dt
            )
            x_ref_val = run_simulation.generate_mpc_reference_trajectory_world(
                quad_world_state, platform_traj_pred, self.N
            )

            # 步骤 3.3: 构建当前步的代价函数参数
            Q_nlp_val = np.concatenate([
                np.zeros(self.nx),                  # X_0 (初始状态) 的代价为0
                np.tile(2 * self.q_weights, self.N),     # X_1 到 X_N 的状态代价
                np.tile(2 * self.r_weights, self.N)      # U_0 到 U_{N-1} 的控制代价
            ])

            p_nlp_list = [np.zeros(self.nx)] # 初始状态x0无线性代价
            for k in range(self.N):
                # 使用向量进行元素级乘法，等效于 Q @ x_ref，但效率更高
                p_nlp_list.append(-2 * self.q_weights * x_ref_val[:, k])
            p_nlp_list.append(np.zeros(self.nu * self.N)) # 控制量 u 无线性代价
            p_nlp_val = np.concatenate(p_nlp_list)

            # 步骤 3.4: 调用MPC求解器获取最优控制输入
            u_opt_quad = self.mpc_solver.solve(quad_world_state, Q_nlp_val, p_nlp_val)
            print(f"最优控制输入: {u_opt_quad}")

            # 步骤 3.5: 将控制指令应用于环境，并执行一步仿真
            action_quad=flu_normalized_to_frd_omega(u_opt_quad[1:])
            print(f"转换后角速度: {action_quad}")

            self.publish_rates_setpoint(u_opt_quad[0],action_quad[0],action_quad[1],action_quad[2])
            
            action = {'quadrotor': quad_world_state, 'platform': self.platform_control}
            rel_obs, self.obs, terminated, truncated, info = self.env.step(action)
            
            # 步骤 3.6: 记录当前步的数据
            self.history['time'].append(self.step * self.env.dt)
            self.history['quad_pos'].append(info['quadrotor']['position'])
            self.history['quad_vel'].append(info['quadrotor']['velocity'])
            self.history['quad_quat'].append(info['quadrotor']['quaternions'])
            self.history['plat_pos'].append(info['platform']['position'])
            self.history['plat_vel'].append(info['platform']['velocity'])
            self.history['plat_psi'].append(info['platform']['psi']) 
            self.history['rel_pos'].append(rel_obs[:3])
            self.history['control_input'].append(u_opt_quad)

            self.step+=1


            if terminated or truncated  :

                if info["success"]==True:
                    self.get_logger().info('降落成功，任务完成！')
                    self.land()
                    self.disarm()                
                else:
                    self.land()
                    self.get_logger().info('降落失败，切换到自动降落模式！')
                for key, val in self.history.items():
                    self.history[key] = np.array(val)

                    # --- 4. 生成并保存结果 ---
                output_directory = "simulation_results_world_frame"
                run_simulation.plot_results(self.history, output_directory)
                run_simulation.create_animation(self.history, output_directory)
                print(f"\n测试完成！所有结果已保存在 '{output_directory}' 文件夹中。")
                exit(0)

        if self.offboard_setpoint_counter < 11:
            self.offboard_setpoint_counter += 1


def main(args=None) -> None:

    # --- 1. 模拟参数 ---
    simulation_params = {
        'quad_init_position': np.array([0.0, 0.0, 5.0]),
        'quad_init_velocity': np.array([0.0, 0.0, 0.0]),
        'quad_init_quaternions': run_simulation.euler_to_quaternion(0, 0, np.deg2rad(0)),
        
        'platform_init_state': np.array([5.0, 5.0, 0.8, np.deg2rad(30)]),
        'platform_u1': 0.2,
        'platform_u2': np.deg2rad(10)
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
