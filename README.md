<div align="center">

<h1>Single-Inverted-Pendulum 学习记录</h1>
<h1>Single Inverted Pendulum Learning Records</h1>

<p><strong>🎯 项目概述 / Project Overview</strong></p>
<p><i>1. 强化学习Q-learning算法解决Gym的Cartpole-v1模型</i></p>
<p><i>Q-learning reinforcement learning algorithm for solving Gym's Cartpole-v1 model</i></p>
<p><i>2. 双环PID控制一阶倒立摆Mujoco仿真</i></p>
<p><i>Double-loop PID control for single inverted pendulum Mujoco simulation</i></p>

<p><i>具体代码及README文件见各项目文件夹</i></p>
<p><i>For detailed code and README files, see respective project folders</i></p>

</div>

---

# 1️⃣ Q-Learning 强化学习 / Q-Learning Reinforcement Learning

使用 Q-learning 强化学习训练 OpenAI Gym 中的经典平衡控制任务 **CartPole-v1**。

*Training the classic balance control task **CartPole-v1** from OpenAI Gym using Q-learning reinforcement learning.*

**780 Episode 后 Cartpole 表现 / CartPole Performance After 780 Episodes:**

<p align="center">
  <img src="Photos/Cartpole.gif" width="400"/>
</p>

## 🟠 环境简介 / Environment Introduction

CartPole 是一个一阶倒立摆环境，其任务是通过移动小车来保持杆子竖直。

*CartPole is a single inverted pendulum environment where the task is to keep the pole upright by moving the cart.*

将该游戏的复杂情景在进行强化学习时抽象为三个变量：

*The complex scenario of this game is abstracted into three variables for reinforcement learning:*

**状态（State）**、**动作（Action）** 和 **奖励（Reward）**

***State**, **Action**, and **Reward***

---

### 状态 / State

每个环境状态由以下 **4 个连续变量** 组成，表示当前物理系统的信息：

*Each environment state consists of the following **4 continuous variables** representing the current physical system information:*

| 特征名称<br>*Feature Name* | 含义<br>*Description* | 数值范围<br>*Value Range* | 离散方式<br>*Discretization* |
|:--------------------:|:----------------------------:|:------------------:|:------------:|
| 小车位置 `cart_pos`<br>*Cart Position* | 小车相对于中心的水平位置<br>*Cart's horizontal position relative to center* | [-2.4, 2.4] | 分为 4 段<br>*4 segments* |
| 小车速度 `cart_v`<br>*Cart Velocity* | 小车的水平移动速度<br>*Cart's horizontal velocity* | [-3.0, 3.0] | 分为 4 段<br>*4 segments* |
| 杆子角度 `pole_angle`<br>*Pole Angle* | 杆子与竖直方向的偏移角度<br>*Pole's deviation angle from vertical* | [-0.5, 0.5] (rad) | 分为 4 段<br>*4 segments* |
| 杆子角速度 `pole_v`<br>*Pole Angular Velocity* | 杆子旋转的角速度<br>*Angular velocity of pole rotation* | [-2.0, 2.0] | 分为 4 段<br>*4 segments* |

对上述变量进行离散化，最终共有：

*After discretizing the above variables, there are a total of:*

> 4 × 4 × 4 × 4 = 256 种可能状态 / *256 possible states*

---

### 动作 / Action

动作空间中，每个时间步只能执行以下两种动作之一：

*In the action space, only one of the following two actions can be executed at each time step:*

| 动作编号<br>*Action ID* | 动作含义<br>*Action Description* |
|:--------:|:------------:|
| `0` | 向左施加力<br>*Apply force left* |
| `1` | 向右施加力<br>*Apply force right* |

算法会根据当前状态的 Q 值表选择一个动作来与游戏环境交互。

*The algorithm selects an action based on the Q-value table of the current state to interact with the game environment.*

---

### 奖励 / Reward

环境的奖励机制如下：

*The environment's reward mechanism is as follows:*

- 每撑过一个时间步（不失败）奖励 `+1`
  
  *Reward `+1` for each time step survived (without failure)*

- 若提前失败（杆子倾倒或小车出界），立即给予 `-200` 的惩罚奖励
  
  *If failure occurs early (pole falls or cart goes out of bounds), immediately give `-200` penalty reward*

## 🟠 Q-Learning 公式（更新Q表）/ Q-Learning Formula (Q-Table Update)

Q-learning 算法通过以下公式不断更新状态-动作函数 Q(s, a)：

*Q-learning algorithm continuously updates the state-action function Q(s, a) through the following formula:*

$$
Q(s_t, a_t) \leftarrow (1 - \alpha) \cdot Q(s_t, a_t) + \alpha \cdot \left[ r_t + \gamma \cdot \max_{a'} Q(s_{t+1}, a') \right]
$$

### 符号含义说明 / Symbol Definitions

| 符号<br>*Symbol* | 含义说明<br>*Description* |
|:-----------------------:|:--------------------------------------------------:|
| `Q(s, a)` | 当前状态 `s` 下，执行动作 `a` 的 Q 值<br>*Q-value for executing action `a` in current state `s`* |
| `α` (alpha) | 学习率，控制新旧信息的更新比例<br>*Learning rate, controlling the update ratio of new/old information* |
| `r` | 当前步获得的奖励<br>*Reward obtained at current step* |
| `γ` (gamma) | 折扣因子，衡量未来奖励的重要性<br>*Discount factor, measuring the importance of future rewards* |
| `max_a' Q(s', a')` | 下一个状态 `s'` 下所有可能动作中的最大 Q 值<br>*Maximum Q-value among all possible actions in next state `s'`* |

此更新规则将当前的 Q 表，通过不断尝试和更新，最终学得最优策略。

*This update rule allows the current Q-table to eventually learn the optimal policy through continuous trial and update.*

## 🟠 控制策略：ε-贪婪 / Control Strategy: ε-Greedy

为平衡探索与利用，采用 ε-贪婪策略：

*To balance exploration and exploitation, an ε-greedy strategy is adopted:*

- **ε** 随着训练进程逐步衰减 / ***ε** gradually decays with training progress*:

$$
ε = 0.5 × (0.99^n), \quad (n \text{ 为 Episode / is Episode})
$$

- 在每次决策中，以 `1-ε` 的概率选择当前 Q 表中最优动作，以 `ε` 的概率随机选择动作。

  *In each decision, select the optimal action from the current Q-table with probability `1-ε`, and select a random action with probability `ε`.*

## 🟠 学习过程、成功条件与退出机制 / Learning Process, Success Criteria & Exit Mechanism

- **学习率 / Learning Rate**：`α = 0.2`
- **折扣因子 / Discount Factor**：`γ = 0.99`
- 每局游戏最多 `200` 步 / *Maximum `200` steps per game*
- 总训练局数为 `1000` 局 / *Total training episodes: `1000`*
- 当连续 100 局游戏的平均得分达到或超过 `195` 时，训练提前终止，并输出成功提示信息

  *When the average score of 100 consecutive games reaches or exceeds `195`, training terminates early with success notification*

---

# 2️⃣ 双环PID控制Mujoco仿真 / Double-loop PID Control with Mujoco Simulation

使用双环（角度+位置）PID 控制器与 DeepMind Control Suite 的 Mujoco 倒立摆模型实现仿真

*Implementing simulation using double-loop (angle + position) PID controller with DeepMind Control Suite's Mujoco inverted pendulum model*

调整双闭环PID控制器参数后Cartpole表现：

*CartPole performance after tuning double closed-loop PID controller parameters:*

<p align="center">
  <img src="https://raw.githubusercontent.com/SelfGala/Inverted-Pendulum/main/Photos/dm_control_cartpole.gif" width="400"/>
</p>

## 🟠 数学建模：一阶倒立摆动力学模型 / Mathematical Modeling: Single Inverted Pendulum Dynamics

以标准的 **一阶倒立摆（Single Inverted Pendulum on a Cart）** 为研究对象，下图为**参考一阶倒立摆**模型：

*Taking the standard **Single Inverted Pendulum on a Cart** as the research object, the figure below shows the **reference single inverted pendulum** model:*

<p align="center">
  <img src="Photos/Cartpole-math-model.jpg" width="500"/>
</p>

### 系统参数 / System Parameters

| 符号<br>*Symbol* | 含义<br>*Description* | 单位<br>*Unit* | dm_control-xml模型参数<br>*dm_control-xml Model Parameters* |
|:----------:|:------------------:|:-------------:|:---------------------:|
| `L` | 摆杆长度<br>*Pendulum length* | 米 (m) | 0.6 |
| `m` | 摆杆质量<br>*Pendulum mass* | 千克 (kg) | 4.2 |
| `M` | 小车质量<br>*Cart mass* | 千克 (kg) | 8 |
| `g` | 重力加速度<br>*Gravitational acceleration* | 米/秒² (m/s²) | -9.81 |
| `b` | 摩擦阻尼系数<br>*Friction damping coefficient* | 牛·秒/米 (N·s/m) | 0.1 |

### 系统状态变量 / System State Variables

| 符号<br>*Symbol* | 含义<br>*Description* | 单位<br>*Unit* |
|:-----------------:|:--------------------------:|:----------------:|
| `x` | 小车位置<br>*Cart position* | 米 (m) |
| `𝑥̇` (x_dot) | 小车速度<br>*Cart velocity* | 米/秒 (m/s) |
| `θ` (theta) | 摆杆偏离竖直的角度<br>*Pole deviation angle from vertical* | 弧度 (rad) |
| `θ̇` (theta_dot) | 摆杆角速度<br>*Pole angular velocity* | 弧度/秒 (rad/s) |

其动力学方程可由拉格朗日方程推导：

*The dynamics equations can be derived from Lagrangian equations:*

**水平方向 / Horizontal Direction** 运动方程 / *Motion Equation*：

$$
F = (M + m)·ẍ + b·ẋ + m·l·θ̈·cos(θ) − m·l·(θ̇)²·sin(θ)
$$

**竖直方向 / Vertical Direction** 运动方程 / *Motion Equation*：

$$
P - m·g = -m·l·θ̈·sin(θ) - m·l·(θ̇)²·cos(θ)
$$

对两个运动方程进行近似处理、线性化处理，cos(θ)≈1，sin(θ)≈θ；再进行拉普拉斯变换，得到：

*After approximation and linearization of the two motion equations, where cos(θ)≈1, sin(θ)≈θ, and applying Laplace transform:*

<p align="center">
  <img src="Photos/Laplace.png" width="350"/>
</p>

由拉普拉斯变换解出两个方向的传递函数，控制系统的状态空间方程可写成如下形式：

*From the Laplace transform, the transfer functions in both directions are solved, and the state space equation of the control system can be written as:*

$$
ẋ=AX+Bu
$$

$$
Y=CX+Du
$$

> u表示系统控制输入向量，x表示系统状态变量，y表示系统的输出向量，A表示系统的状态矩阵，B表示系统控制输入矩阵，C表示系统输出观测矩阵，D表示系统输入输出矩阵。
> 
> *u represents the system control input vector, x represents the system state variables, y represents the system output vector, A represents the system state matrix, B represents the system control input matrix, C represents the system output observation matrix, D represents the system input-output matrix.*

根据运动方程组和拉普拉斯变换对ẍ和θ̈求解可得解如下：

*According to the motion equation system and Laplace transform, solving for ẍ and θ̈ yields the following solutions:*

<p align="center">
  <img src="Photos/Solution_X.png" width="500"/>
</p>

<p align="center">
  <img src="Photos/Solution_Y.png" width="250"/>
</p>

## 🟠 PID控制器设计 / PID Controller Design

PID控制器的参数`Kp,Ki,Kd`通过经验数据或试凑法很难进行调整得到合适的控制器，我们在得到传递函数之后，可以通过matlab的`pidtune`函数来对参数进行一个简单的设计。

*The PID controller parameters `Kp, Ki, Kd` are difficult to adjust through empirical data or trial-and-error methods to obtain a suitable controller. After obtaining the transfer function, we can use MATLAB's `pidtune` function for simple parameter design.*

例如我们根据dm_control的cartpole-xml模型的物理参数可以得到其角度的传递函数：

*For example, based on the physical parameters of the dm_control cartpole-xml model, we can obtain its angle transfer function:*

$$
P_θ = θ_s / U_s = \frac{0.276·s²}{s⁴ + 0.11·s³ − 33.06·s² − 2.694·s}
$$

对其应用`pidtune`: [Kpid, ~, ~] = pidtune(P_angle, 'PID')，我们可以得到：

*Applying `pidtune`: [Kpid, ~, ~] = pidtune(P_angle, 'PID'), we obtain:*

- K_p = 449.4918
- K_i = 801.4594
- K_d = 56.3668

通过构建仿真闭环系统的bode图、单位阶跃响应、单位脉冲响应，我们可以看到在一定时间后，输出已经收敛：

*By constructing the Bode plot, unit step response, and unit impulse response of the simulated closed-loop system, we can see that the output has converged after a certain time:*

<p align="center">
  <img src="Photos/Bode_angle.png" width="1100"/>
</p>

## 🟠 dm_control Mujoco仿真设计 / dm_control Mujoco Simulation Design

我们应用Deepmind构建的Cartpole模型来对倒立摆进行仿真，具体xml模型文件请见文件夹，其中需要注意的是xml默认质量根据物体体积计算，默认密度为水的密度：1000kg/m³

*We use the CartPole model built by DeepMind for inverted pendulum simulation. For specific xml model files, please see the folder. Note that xml default mass is calculated based on object volume, with default density being water density: 1000kg/m³*

dm_control-Cartpole默认引用方法具体见文件夹，模型参数说明见下：

*For dm_control-CartPole default usage methods, see the folder. Model parameter descriptions are as follows:*

### 模型结构摘要 / Model Structure Summary

| 元素<br>*Element* | 名称/属性<br>*Name/Attribute* | 类型<br>*Type* | 说明<br>*Description* |
|------------|------------------|--------------|----------------------------------|
| `cart` | Joint: `slider` | slide joint | 小车沿 X 方向滑动，范围 ±1 m<br>*Cart slides along X direction, range ±1 m* |
| `pole` | Joint: `hinge` | hinge joint | 摆杆绕 Y 轴旋转（垂直平面内）<br>*Pole rotates around Y axis (in vertical plane)* |
| `floor` | Geom | plane | 地面（静态）<br>*Ground (static)* |
| `cpole` | Geom | capsule | 摆杆形状，0.6m 长度<br>*Pole shape, 0.6m length* |
| `cart` | Geom | box | 小车主体，长 0.2m 宽 0.1m 高 0.05m<br>*Cart body, 0.2m × 0.1m × 0.05m* |
| `mocap1` / `mocap2` | Body | mocap | 可视参考物体，不参与控制<br>*Visual reference objects, not involved in control* |

---

### 控制与执行机制 / Control and Actuation Mechanism

| 组件类型<br>*Component Type* | 名称<br>*Name* | 属性<br>*Attribute* | 说明<br>*Description* |
|---------------|--------------|------------------|------------------------------|
| `actuator` | `slide` | motor | 控制小车（作用于 slider 关节）<br>*Controls cart (acts on slider joint)* |
|  | gear = `50` |  | 增益放大倍数<br>*Gain amplification factor* |
|  | ctrlrange = `[-1, 1]` |  | 控制范围（实际力：±50N）<br>*Control range (actual force: ±50N)* |
| `sensor` | `accelerometer` | on `cart sensor` | 小车加速度<br>*Cart acceleration* |
|  | `touch` | on `cart sensor` | 碰撞检测<br>*Collision detection* |

## 🟠 致谢与引用 / Acknowledgments and Citations

本项目基于 DeepMind 开源的物理引擎平台 [dm_control](https://github.com/google-deepmind/dm_control) 构建，使用其中的 `cartpole` 环境进行倒立摆仿真和控制器测试。

*This project is built based on DeepMind's open-source physics engine platform [dm_control](https://github.com/google-deepmind/dm_control), using its `cartpole` environment for inverted pendulum simulation and controller testing.*

感谢该项目提供了高精度的 MuJoCo 封装接口与标准任务集。

*Thanks to this project for providing high-precision MuJoCo wrapper interfaces and standard task sets.*

仓库地址 / *Repository*：[https://github.com/google-deepmind/dm_control](https://github.com/google-deepmind/dm_control)

---

<div align="center">
<p><strong>🔬 技术栈 / Tech Stack</strong></p>
<p><code>Python</code> • <code>OpenAI Gym</code> • <code>Q-Learning</code> • <code>PID Control</code> • <code>MuJoCo</code> • <code>dm_control</code> • <code>MATLAB</code></p>
</div>
