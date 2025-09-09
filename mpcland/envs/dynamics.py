"""
动力学模型文件 - 包含四旋翼和移动平台的数学模型
"""
from dataclasses import dataclass
import numpy as np
from math import sin, cos, tan, atan
import config.config as Config

# =============== 状态的数据结构定义 ===============
# 使用dataclass可以方便地创建带有类型提示的、结构化的数据容器。

@dataclass
class QuadrotorState:
    def __init__(self, position=None, velocity=None, quaternions=None):

        """封装四旋翼无人机的状态向量。"""
        self.position = np.array(position) # 世界坐标系下的位置 [x, y, z]
        self.velocity = np.array(velocity)# 世界坐标系下的速度 [vx, vy, vz]
        self.quaternions = np.array(quaternions) # 世界坐标系下的姿态四元数 [qw, qx, qy, qz]

    def reset(self, position, velocity, quaternions):
        """初始化状态向量。"""
        self.position = np.array(position)
        self.velocity = np.array(velocity)
        self.quaternions = np.array(quaternions)

    def copy(self):
        """创建当前状态的深拷贝，避免在传递中意外修改原始数据。"""
        return QuadrotorState(
            position=self.position.copy(),
            velocity=self.velocity.copy(),
            quaternions=self.quaternions.copy()
        )

@dataclass
class PlatformState:
    """封装移动平台的状态向量（基于自行车模型）。"""
    x: float      # 世界坐标系下的x坐标
    y: float      # 世界坐标系下的y坐标
    v: float      # 平台自身的速度大小 (m/s)
    psi: float    # 平台的偏航角 (rad)，即车头朝向

    def copy(self):
        """创建当前状态的深拷贝。"""
        return PlatformState(x=self.x, y=self.y, v=self.v, psi=self.psi)

# =============== 四元数运算辅助函数 ===============

def quaternion_multiply(q1, q2):
    """
    执行四元数乘法 q_new = q1 * q2。
    这在组合旋转时非常有用。
    """
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    return np.array([w, x, y, z])

def rotate_vector_by_quaternion(q, v):
    """
    使用四元数 q 旋转一个三维向量 v。
    数学公式为: v_rotated = q * v * q_conjugate
    """
    # 将向量 v 提升为一个纯四元数 (w=0)
    q_v = np.array([0, v[0], v[1], v[2]])
    # 计算四元数 q 的共轭
    q_conj = np.array([q[0], -q[1], -q[2], -q[3]])
    # 执行旋转计算
    result_quat = quaternion_multiply(quaternion_multiply(q, q_v), q_conj)
    # 返回旋转后四元数的向量部分
    return result_quat[1:4]


# =============== 移动平台动力学模型 ===============

class MovingPlatformDynamics:
    """模拟移动平台的运动（自行车模型）。"""
    def __init__(self):
        """从配置文件加载平台参数。"""
        self.l_f = Config.MovingPlatform.L_F
        self.l_r = Config.MovingPlatform.L_R
        self.v_max = Config.MovingPlatform.V_MAX
        self.state: PlatformState = None

    def _beta(self, u2: float):
        """计算车辆的侧滑角 (beta)。"""
        return atan(tan(u2) * self.l_r / (self.l_f + self.l_r))

    def compute_derivatives(self, state: PlatformState, u1: float, u2: float):
        """
        计算平台状态的时间导数。
        Args:
            state: 当前的平台状态。
            u1: 纵向加速度控制输入。
            u2: 前轮转向角控制输入。
        Returns:
            一个元组 (dx, dy, dv, dpsi)，分别是各状态量的导数。
        """
        beta_val = self._beta(u2)
        v, psi = state.v, state.psi

        dx = v * cos(psi + beta_val)
        dy = v * sin(psi + beta_val)
        dv = u1
        dpsi = (v / self.l_r) * sin(beta_val)

        return dx, dy, dv, dpsi

    def step(self, control: dict, dt: float):
        """使用前向欧拉法，将平台状态推进一个时间步 dt。"""
        if self.state is None:
            raise ValueError("平台状态未初始化，请先调用 reset()")

        u1 = control['u1']  # 纵向加速度
        u2 = control['u2']  # 前轮转向角

        # 计算导数
        dx, dy, dv, dpsi = self.compute_derivatives(self.state, u1, u2)

        # 欧拉积分更新状态
        new_v = self.state.v + dv * dt
        new_v = np.clip(new_v, 0, self.v_max)  # 限制速度不小于0且不超过最大值

        self.state = PlatformState(
            x=self.state.x + dx * dt,
            y=self.state.y + dy * dt,
            v=new_v,
            psi=self.state.psi + dpsi * dt
        )

    def reset(self, init_state: np.ndarray):
        """重置平台到指定的初始状态。"""
        self.state = PlatformState(
            x=init_state[0],
            y=init_state[1],
            v=init_state[2],
            psi=init_state[3]
        )
