"""
MPC控制器完整仿真与可视化程序 (世界坐标系规划)

=================================================================================
程序目的:
    本脚本用于对基于世界坐标系规划的MPC控制器进行完整的闭环仿真。它在一个预设的
    场景下运行无人机着陆任务，记录整个过程中的详细数据，并最终生成直观的分析
    图表和3D动画，以全面评估控制器的性能。

核心控制策略:
    1. 获取无人机和移动平台在统一世界坐标系下的绝对状态。
    2. 基于平台的动力学模型，在世界坐标系中预测其未来N步的运动轨迹。
    3. 生成一个同样在世界坐标系下的无人机参考轨迹，该轨迹旨在引导无人机平滑地
       拦截平台预测的轨迹。
    4. 将无人机的世界坐标系状态作为初始条件，将世界坐标系参考轨迹作为目标，
       送入MPC求解器进行优化，并在每个时间步重复此过程。

适用场景:
    - 对控制器进行端到端的性能验证，而不仅仅是单步决策。
    - 分析系统的暂态和稳态响应，如跟踪误差、收敛速度等。
    - 生成用于报告、演示或进一步分析的可视化结果（图表、动画）。
=================================================================================
"""
import numpy as np
from math import cos, sin
import os
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
from tqdm import tqdm

# 从项目中导入必要的模块
from envs import  MovingPlatformDynamics, PlatformState
import config.config as Config

# 设置matplotlib以正确显示中文和负号(linux系统只能用英文)
# plt.rcParams['font.sans-serif'] = ['SimHei']
# plt.rcParams['axes.unicode_minus'] = False

# ==============================================================================
# 辅助函数 
# ==============================================================================

def euler_to_quaternion(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """将欧拉角 (roll, pitch, yaw) 转换为四元数 (qw, qx, qy, qz)。"""
    cy, sy = cos(yaw * 0.5), sin(yaw * 0.5)
    cp, sp = cos(pitch * 0.5), sin(pitch * 0.5)
    cr, sr = cos(roll * 0.5), sin(roll * 0.5)
    qw = cr * cp * cy + sr * sp * sy
    qx = sr * cp * cy - cr * sp * sy
    qy = cr * sp * cy + sr * cp * sy
    qz = cr * cp * sy - sr * sp * cy
    return np.array([qw, qx, qy, qz])

def calc_vz_ref(current_relative_z: float) -> float:
    """根据当前相对高度，分阶段计算参考下降速度。"""
    if current_relative_z > 1.5:
        return -1.0
    elif current_relative_z > 0.5:
        return -0.5
    else:
        return -0.2

# ==============================================================================
# 核心算法模块 
# ==============================================================================

def predict_platform_trajectory_world(
    current_platform_state: PlatformState, 
    platform_control: np.ndarray, 
    N: int, 
    dt: float
) -> dict:
    """
    在世界坐标系中，使用平台的动力学模型预测其未来N步的轨迹。
    这是实现前瞻性跟踪的关键步骤。
    """
    predictor = MovingPlatformDynamics()
    predictor.state = current_platform_state.copy()
    pred_pos, pred_vel, pred_psi = [], [], []
    control_dict = {'u1': platform_control[0], 'u2': platform_control[1]}

    for _ in range(N):
        predictor.step(control_dict, dt)
        state = predictor.state
        pred_pos.append([state.x, state.y, Config.MovingPlatform.HEIGHT])
        vx_world = state.v * cos(state.psi)
        vy_world = state.v * sin(state.psi)
        pred_vel.append([vx_world, vy_world, 0.0])
        pred_psi.append(state.psi)

    return {
        'pos': np.array(pred_pos),
        'vel': np.array(pred_vel),
        'psi': np.array(pred_psi)
    }

def generate_mpc_reference_trajectory_world(
    quadrotor_world_state: np.ndarray, 
    platform_trajectory_prediction: dict, 
    N: int
) -> np.ndarray:
    """
    在世界坐标系中，为MPC生成优化的无人机参考轨迹。
    核心思想是设计一条轨迹，使无人机能平滑地消除与平台预测轨迹之间的初始误差。
    """
    nx = 10
    x_ref = np.zeros((nx, N))
    current_quad_pos = quadrotor_world_state[:3]
    current_platform_pos_approx = platform_trajectory_prediction['pos'][0]
    initial_pos_error = current_quad_pos - current_platform_pos_approx

    for k in range(N):
        convergence_factor = 1.0 - (k + 1) / (N + 1)
        p_ref_k = platform_trajectory_prediction['pos'][k] + initial_pos_error * convergence_factor
        
        v_correction = -initial_pos_error / (N * Config.DELTA_T) * 1.5
        v_ref_k = platform_trajectory_prediction['vel'][k] + v_correction * convergence_factor
        
        # 使用无人机在第k步的【参考高度】来计算相对高度，这比使用当前高度更具前瞻性。
        relative_z_k = p_ref_k[2] - platform_trajectory_prediction['pos'][k][2]
        v_ref_k[2] = calc_vz_ref(relative_z_k)
        
        psi_plat_k = platform_trajectory_prediction['psi'][k]
        q_ref_k = euler_to_quaternion(0, 0, psi_plat_k)
        
        x_ref[:, k] = np.concatenate([p_ref_k, v_ref_k, q_ref_k])

    return x_ref

# ==============================================================================
# 可视化模块
# ==============================================================================

def plot_results(history: dict, output_dir: str):
    """生成并保存仿真过程中的关键状态图表。"""
    print("--- 正在生成静态图表 ---")
    os.makedirs(output_dir, exist_ok=True)
    time_ax = history['time']
    
    # 图1: 3D轨迹图
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    quad_p, plat_p = history['quad_pos'], history['plat_pos']
    ax.plot(quad_p[:, 0], quad_p[:, 1], quad_p[:, 2], label='Drone trajectory', color='b')
    ax.plot(plat_p[:, 0], plat_p[:, 1], plat_p[:, 2], label='Platform trajectory', color='r', linestyle='--')
    ax.scatter(quad_p[0, 0], quad_p[0, 1], quad_p[0, 2], c='blue', s=50, marker='o', label='Drone starting point')
    ax.scatter(quad_p[-1, 0], quad_p[-1, 1], quad_p[-1, 2], c='blue', s=80, marker='*', label='Drone endpoint')
    ax.scatter(plat_p[0, 0], plat_p[0, 1], plat_p[0, 2], c='red', s=50, marker='o', label='Platform Starting Point')
    ax.set_xlabel('X (m)'), ax.set_ylabel('Y (m)'), ax.set_zlabel('Z (m)')
    ax.set_title('3D trajectory comparison'), ax.legend(), ax.grid(True)
    ax.set_aspect('equal', 'box')
    plt.savefig(os.path.join(output_dir, '1_trajectory_3d.png'))
    plt.close(fig)

    # 图2: 相对位置与无人机速度
    fig, axs = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    rel_p, quad_v = history['rel_pos'], history['quad_vel']
    axs[0].plot(time_ax, rel_p[:, 0], label='Relative Position x')
    axs[0].plot(time_ax, rel_p[:, 1], label='Relative Position y')
    axs[0].plot(time_ax, rel_p[:, 2], label='Relative Position z')
    axs[0].axhline(y=Config.Termination.SUCCESS_XY_ERR_MAX, color='g', linestyle='--', label='Horizontal success boundary')
    axs[0].axhline(y=-Config.Termination.SUCCESS_XY_ERR_MAX, color='g', linestyle='--')
    axs[0].set_ylabel('Relative position (m)'), axs[0].set_title('Time-Varying Relative position'), axs[0].legend(), axs[0].grid(True)
    
    axs[1].plot(time_ax, quad_v[:, 0], label='Drone vx')
    axs[1].plot(time_ax, quad_v[:, 1], label='Drone vy')
    axs[1].plot(time_ax, quad_v[:, 2], label='Drone vz')
    axs[1].set_xlabel('time (s)'), axs[1].set_ylabel('Drone World (m/s)')
    axs[1].set_title('Time-Varying Drone Velocity'), axs[1].legend(), axs[1].grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '2_relative_and_velocity_states.png'))
    plt.close(fig)

    print(f"静态图表已保存至: {output_dir}")


def create_animation(history: dict, output_dir: str, filename="simulation_animation.gif"):
    """创建并保存整个着陆过程的3D动画。"""
    print("--- 正在创建动画 (这可能需要一些时间) ---")
    os.makedirs(output_dir, exist_ok=True)
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    quad_p, plat_p = history['quad_pos'], history['plat_pos']
    
    padding = 2.0
    all_x = np.concatenate((quad_p[:, 0], plat_p[:, 0]))
    all_y = np.concatenate((quad_p[:, 1], plat_p[:, 1]))
    ax.set_xlim(all_x.min() - padding, all_x.max() + padding)
    ax.set_ylim(all_y.min() - padding, all_y.max() + padding)
    ax.set_zlim(0, quad_p[:, 2].max() + padding)
    ax.set_xlabel('X (m)'), ax.set_ylabel('Y (m)'), ax.set_zlabel('Z (m)')
    ax.set_aspect('equal', 'box')

    quad_traj, = ax.plot([], [], [], 'b-', label='Drone trajectory')
    plat_traj, = ax.plot([], [], [], 'r--', label='Platform trajectory')
    quad_pos, = ax.plot([], [], [], 'bo', markersize=8, label='Drone')
    plat_surface, = ax.plot([], [], [], 'g-', linewidth=5, label='Platform surface')
    
    l = Config.Termination.SUCCESS_XY_ERR_MAX
    corners = np.array([[l, -l], [l, l], [-l, l], [-l, -l], [l, -l]])

    def init():
        ax.legend()
        return quad_traj, plat_traj, quad_pos, plat_surface

    def update(i):
        quad_traj.set_data_3d(quad_p[:i + 1, 0], quad_p[:i + 1, 1], quad_p[:i + 1, 2])
        plat_traj.set_data_3d(plat_p[:i + 1, 0], plat_p[:i + 1, 1], plat_p[:i + 1, 2])
        quad_pos.set_data_3d([quad_p[i, 0]], [quad_p[i, 1]], [quad_p[i, 2]])
        
        px, py, pz = history['plat_pos'][i]
        psi = history['plat_psi'][i]
        R = np.array([[cos(psi), -sin(psi)], [sin(psi), cos(psi)]])
        rotated_corners = corners @ R.T + np.array([px, py])
        plat_surface.set_data_3d(rotated_corners[:, 0], rotated_corners[:, 1], [pz] * 5)
        ax.set_title(f'time: {history["time"][i]:.1f}s')
        return quad_traj, plat_traj, quad_pos, plat_surface

    filepath = os.path.join(output_dir, filename)
    anim = FuncAnimation(fig, update, frames=len(history['time']), init_func=init, blit=False, interval=50)
    with tqdm(total=len(history['time']), desc="正在保存GIF") as pbar:
        anim.save(filepath, writer='pillow', fps=20, progress_callback=lambda i, n: pbar.update(1))
    plt.close(fig)
    print(f"动画已保存至: {filepath}")