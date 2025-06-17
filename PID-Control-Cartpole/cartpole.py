import numpy as np
import matplotlib.pyplot as plt
from dm_control import suite
from dm_control import viewer
import time

class PIDController:
    """PID控制器类"""
    def __init__(self, kp, ki, kd, dt=0.02):
        self.kp = kp  # 比例增益
        self.ki = ki  # 积分增益
        self.kd = kd  # 微分增益
        self.dt = dt  # 时间步长
        
        # 内部状态
        self.prev_error = 0
        self.integral = 0
        
    def update(self, error):
        """更新PID控制器并返回控制输出"""
        # 比例项
        proportional = self.kp * error
        
        # 积分项
        self.integral += error * self.dt
        integral_term = self.ki * self.integral
        
        # 微分项
        derivative = (error - self.prev_error) / self.dt
        derivative_term = self.kd * derivative
        
        # 总输出
        output = proportional + integral_term + derivative_term
        
        # 更新前一次误差
        self.prev_error = error
        
        return output
    
    def reset(self):
        """重置控制器状态"""
        self.prev_error = 0
        self.integral = 0

class CartpolePIDController:
    """倒立摆PID控制器"""
    def __init__(self):
        # PID参数
        # 角度控制器（主要控制器）
        self.angle_pid = PIDController(kp= , ki= , kd= )
        
        # 位置控制器（辅助控制器）
        self.position_pid = PIDController(kp= , ki= , kd= )
        
        # 控制权重
        self.angle_weight = 1     # 角度控制权重
        self.position_weight =  # 位置控制权重
        
        # 目标值
        self.target_angle = 0.0      # 目标角度（直立）
        self.target_position = 0.0   # 目标位置（中心）
        
    def get_action(self, observation):
        """根据观测值计算控制动作"""
        # 从观测值中提取状态
        cart_position = observation[0]      # 小车位置
        cart_velocity = observation[1]      # 小车速度
        pole_angle = observation[2]         # 摆杆角度
        pole_angular_velocity = observation[3]  # 摆杆角速度
        
        # 计算角度误差（目标是保持直立，即角度为0）
        angle_error = self.target_angle - pole_angle
        
        # 计算位置误差（目标是保持在中心）
        position_error = self.target_position - cart_position
        
        # PID控制
        angle_control = self.angle_pid.update(angle_error)
        position_control = self.position_pid.update(position_error)
        
        # 组合控制信号
        total_control = (self.angle_weight * angle_control + 
                        self.position_weight * position_control)
        
        # 限制控制输出
        max_force = 10.0
        total_control = np.clip(total_control, -max_force, max_force)
        
        return np.array([total_control])
    
    def reset(self):
        """重置控制器"""
        self.angle_pid.reset()
        self.position_pid.reset()

def run_pid_control():
    """运行PID控制的主函数"""
    # 创建环境
    env = suite.load(domain_name="cartpole", task_name="balance")
    
    # 创建PID控制器
    controller = CartpolePIDController()
    
    # 数据记录
    time_steps = []
    angles = []
    positions = []
    controls = []
    
    # 重置环境
    time_step = env.reset()
    time.sleep(0.2)
    controller.reset()
    
    print("开始PID控制...")
    print("目标：保持摆杆直立在中心位置")
    print("按Ctrl+C停止...")
    
    try:
        step_count = 0
        start_time = time.time()
        
        while True:
            # 获取当前观测
            observation = time_step.observation
            
            # 提取状态信息
            cart_pos = observation['position'][0]
            cart_vel = observation['velocity'][0] 
            pole_angle = observation['position'][1]
            pole_vel = observation['velocity'][1]
            
            # 构建状态向量
            state = np.array([cart_pos, cart_vel, pole_angle, pole_vel])
            
            # 计算控制动作
            action = controller.get_action(state)
            
            # 执行动作
            time_step = env.step(action)
            
            # 记录数据
            current_time = time.time() - start_time
            time_steps.append(current_time)
            angles.append(pole_angle)
            positions.append(cart_pos)
            controls.append(action[0])
            
            # 打印状态信息
            if step_count % 50 == 0:  # 每50步打印一次
                print(f"步数: {step_count:4d}, "
                      f"时间: {current_time:6.2f}s, "
                      f"角度: {pole_angle:6.3f} rad ({np.degrees(pole_angle):6.1f}°), "
                      f"位置: {cart_pos:6.3f}, "
                      f"控制: {action[0]:6.2f}")
            
            step_count += 1
            
            # 可选：限制运行时间或步数
            if step_count > 10000:  # 运行5000步后停止
                break
                
            time.sleep(0.01)  # 控制循环频率
            
    except KeyboardInterrupt:
        print("\n用户中断，停止控制...")
    
    # 绘制结果
    plot_results(time_steps, angles, positions, controls)
    
    return time_steps, angles, positions, controls

def plot_results(time_steps, angles, positions, controls):
    """绘制控制结果"""
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    
    # 角度图
    axes[0].plot(time_steps, np.degrees(angles), 'b-', linewidth=2)
    axes[0].axhline(y=0, color='r', linestyle='--', alpha=0.7, label='Target angle')
    axes[0].set_ylabel('Pole angle (°)')
    axes[0].set_title('PID Control')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()
    
    # 位置图
    axes[1].plot(time_steps, positions, 'g-', linewidth=2)
    axes[1].axhline(y=0, color='r', linestyle='--', alpha=0.7, label='Target position')
    axes[1].set_ylabel('Cart position (m)')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    
    # 控制输入图
    axes[2].plot(time_steps, controls, 'orange', linewidth=2)
    axes[2].set_ylabel('Force (N)')
    axes[2].set_xlabel('Time (s)')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def run_with_viewer():
    """带可视化界面运行"""
    env = suite.load(domain_name="cartpole", task_name="balance")
    controller = CartpolePIDController()
    
    def policy(time_step):
        """策略函数，供viewer使用"""
        if time_step.first():
            controller.reset()
            
        observation = time_step.observation
        cart_pos = observation['position'][0]
        cart_vel = observation['velocity'][0] 
        pole_angle = observation['position'][1]
        pole_vel = observation['velocity'][1]
        
        state = np.array([cart_pos, cart_vel, pole_angle, pole_vel])
        action = controller.get_action(state)
        
        return action
    
    # 启动viewer
    viewer.launch(env, policy=policy)

if __name__ == "__main__":
    print("MuJoCo倒立摆PID控制器")
    print("=" * 40)
    print("选择运行模式:")
    print("1. 带数据记录和图表显示")
    print("2. 带可视化界面")
    
    choice = input("请选择 (1 或 2): ").strip()
    
    if choice == "1":
        # 运行并记录数据
        run_pid_control()
    elif choice == "2":
        # 运行带可视化界面
        print("启动可视化界面...")
        run_with_viewer()
    else:
        print("无效选择，使用默认模式（带数据记录）")
        run_with_viewer()