# Double-loop PID simulating with Mujoco

使用双环（角度+位置）PID 控制器与DeepMind Control Suite的Mujoco倒立摆模型实现仿真

调整双闭环PID控制器参数后Cartpole表现：

<p align="center">
  <img src="https://raw.githubusercontent.com/SelfGala/Inverted-Pendulum/main/Photos/dm_control_cartpole.gif" width="400"/>
</p>

## 🟠数学建模：一阶倒立摆动力学模型

以标准的 **一阶倒立摆（Single Inverted Pendulum on a Cart）** 为研究对象，下图为**参考一阶倒立摆**模型：

<p align="center">
  <img src="https://raw.githubusercontent.com/SelfGala/Inverted-Pendulum/main/Photos/Cartpole-math-model.jpg" width="500"/>
</p>

### 系统参数

| 符号     | 含义             | 单位        | dm_control-xml模型参数 |
|:----------:|:------------------:|:-------------:|:---------------------:|
| `L`      | 摆杆长度         | 米 (m)       | 0.6                 |
| `m`      | 摆杆质量         | 千克 (kg)    | 4.2                 |
| `M`      | 小车质量         | 千克 (kg)    | 8                 |
| `g`      | 重力加速度       | 米/秒² (m/s²)| -9.81                |
| `b`      | 摩擦阻尼系数     | 牛·秒/米 (N·s/m) | 0.1              |

### 系统状态变量

| 符号            | 含义                     | 单位           |
|:-----------------:|:--------------------------:|:----------------:|
| `x`             | 小车位置                 | 米 (m)         |
| `𝑥̇` (x_dot)     | 小车速度                 | 米/秒 (m/s)    |
| `θ` (theta)     | 摆杆偏离竖直的角度       | 弧度 (rad)     |
| `θ̇` (theta_dot) | 摆杆角速度               | 弧度/秒 (rad/s)|

其动力学方程可由拉格朗日方程推导：

**水平方向**运动方程：

$$
F = (M + m)·ẍ + b·ẋ + m·l·θ̈·cos(θ) − m·l·(θ̇)²·sin(θ)
$$

**竖直方向**运动方程：

$$
P - m·g = -m·l·θ̈·sin(θ) - m·l·(θ̇)²·cos(θ)
$$

对两个运动方程进行近似处理、线性化处理，cos(θ)=-1，sin(θ)=-Φ；再进行拉普拉斯变换，得到：

<p align="center">
  <img src="https://raw.githubusercontent.com/SelfGala/Inverted-Pendulum/main/Photos/Laplace.png" width="350"/>
</p>

由拉普拉斯变换解出两个方向的传递函数，控制系统的状态空间方程可写成如下形式：

$$
ẋ=AX+Bu
$$

$$
Y=CX+Du
$$

> u表示系统控制输入向量，x表示系统状态变量，y表示系统的输出向量，A表示系统的状态矩阵，B表示系统控制输入矩阵，C表示系统输出观测矩阵，D表示系统输入输出矩阵。

根据运动方程组和拉普拉斯变换对ẍ和Φ̈ 求解可得解如下：

<p align="center">
  <img src="https://raw.githubusercontent.com/SelfGala/Inverted-Pendulum/main/Photos/Solution_X.png" width="500"/>
</p>

<p align="center">
  <img src="https://raw.githubusercontent.com/SelfGala/Inverted-Pendulum/main/Photos/Solution_Y.png" width="250"/>
</p>

## 🟠PID控制器设计

PID控制器的参数`Kp,Ki,Kd`通过经验数据或试凑法很难进行调整得到合适的控制器，我们在得到传递函数之后，可以通过matlab的`pidtune`函数来对参数进行一个简单的设计。

例如，我们根据dm_control的cartpole-xml模型的物理参数可以得到其角度的传递函数：

$$
P_Φ = Φ_s / U_s = (0.276·s²)/(s⁴ + 0.11·s³ − 33.06·s² − 2.694·s)
$$

对其应用`pidtune`: [Kpid, ~, ~] = pidtune(P_bai, 'PID')，我们可以得到：

- K_p= 449.4918
- K_i= 801.4594
- K_d= 56.3668

通过构建仿真闭环系统的bode图、单位阶跃响应、单位脉冲响应，我们可以看到在一定时间后，输出已经收敛：

<p align="center">
  <img src="https://raw.githubusercontent.com/SelfGala/Inverted-Pendulum/main/Photos/Bode_angle.png" width="1100"/>
</p>

## 🟠dm_control Mujoco仿真设计

我们应用Deepmind构建的Cartpole模型来对倒立摆进行仿真，具体xml模型文件请见文件夹，其中需要注意的是xml默认质量根据物体体积计算，默认密度为水的密度：`1000kg/m^3`

dm_control-Cartpole默认引用方法具体见文件夹，模型参数说明见下：

### 模型结构摘要

| 元素       | 名称/属性         | 类型         | 说明                             |
|------------|------------------|--------------|----------------------------------|
| `cart`     | Joint: `slider`  | slide joint  | 小车沿 X 方向滑动，范围 ±1 m     |
| `pole`     | Joint: `hinge`   | hinge joint  | 摆杆绕 Y 轴旋转（垂直平面内）    |
| `floor`    | Geom             | plane        | 地面（静态）                    |
| `cpole`    | Geom             | capsule      | 摆杆形状，0.6m 长度               |
| `cart`     | Geom             | box          | 小车主体，长 0.2m 宽 0.1m 高 0.05m|
| `mocap1` / `mocap2` | Body    | mocap        | 可视参考物体，不参与控制        |

---

### 控制与执行机制

| 组件类型      | 名称         | 属性             | 说明                         |
|---------------|--------------|------------------|------------------------------|
| `actuator`    | `slide`      | motor            | 控制小车（作用于 slider 关节）|
|               | gear = `50`  |                  | 增益放大倍数                  |
|               | ctrlrange = `[-1, 1]` |          | 控制范围（实际力：±50N）     |
| `sensor`      | `accelerometer` | on `cart sensor` | 小车加速度                    |
|               | `touch`      | on `cart sensor` | 碰撞检测                      |
