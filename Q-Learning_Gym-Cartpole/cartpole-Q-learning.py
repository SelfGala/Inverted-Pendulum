import gym
import numpy as np
#Episode Termination:
#        Pole Angle is more than 12 degrees
#        Cart Position is more than 2.4 (center of the cart reaches the edge of the display)
#        Episode length is greater than 200
#Solved Requirements:
#        Considered solved when the average reward is greater than or equal to 195.0 over 100 consecutive trials.


##-------------------------------------------------[1]参数初始化----------------------------------------------------------------
env = gym.make('CartPole-v1', render_mode="human")

max_number_of_steps = 200   # 每一场游戏的最高得分
#---------获胜的条件是最近100场平均得分高于195-------------
goal_average_steps = 195
num_consecutive_iterations = 100
#----------------------------------------------------------
num_episodes = 1000  # 共进行1000场游戏
last_time_steps = np.zeros(num_consecutive_iterations)  # 只存储最近100场的得分

# q_table是一个256*2的二维数组
# 离散化后的状态共有4^4=256中可能的取值，每种状态会对应一个行动
# q_table[s][a]就是当状态为s时作出行动a的有利程度评价值
# 我们的AI模型要训练学习的就是这个映射关系表
q_table = np.random.uniform(low=-1, high=1, size=(4 ** 4, env.action_space.n))


##---------------------------------------------------------[2]自定义函数------------------------------------------------------------
# 分箱处理函数，把[clip_min,clip_max]区间平均分为num段，位于i段区间的特征值x会被离散化为i
def bins(clip_min, clip_max, num):
    return np.linspace(clip_min, clip_max, num + 1)[1:-1]

# 离散化处理，将由4个连续特征值组成的状态矢量转换为一个0~~255的整数离散值
def digitize_state(observation):
    # 将矢量打散回4个连续特征值
    cart_pos, cart_v, pole_angle, pole_v = observation
    # 分别对各个连续特征值进行离散化（分箱处理）
    digitized = [np.digitize(cart_pos, bins=bins(-2.4, 2.4, 4)),    #返回小车位置处于离散化的哪个区间
                 np.digitize(cart_v, bins=bins(-3.0, 3.0, 4)),      #返回小车速度处于离散化的哪个区间
                 np.digitize(pole_angle, bins=bins(-0.5, 0.5, 4)),  #返回竖杆角度处于离散化的哪个区间
                 np.digitize(pole_v, bins=bins(-2.0, 2.0, 4))]      #返回竖杆速度处于离散化的哪个区间
    # 将4个离散值再组合为一个离散值，作为最终结果
    return sum([x * (4 ** i) for i, x in enumerate(digitized)])

# 根据本次的行动及其反馈（下一个时间步的状态），返回下一次的最佳行动
def get_action(state, action, observation, reward, episode):
    next_state = digitize_state(observation)
    epsilon = 0.5 * (0.99 ** episode)  # ε-贪心策略中的ε
    if epsilon <= np.random.uniform(0, 1):
        next_action = np.argmax(q_table[next_state])
    else:
        next_action = np.random.choice([0, 1])
    #  训练学习，更新q_table
    alpha = 0.2     # 学习系数α
    gamma = 0.99    # 报酬衰减系数γ
    q_table[state, action] = (1 - alpha) * q_table[state, action] + alpha * (reward + gamma * q_table[next_state, next_action])

    return next_action, next_state

# 跳出循环函数，训练成功后直接跳出循环
class Getoutofloop(Exception):
    pass

##--------------------------------------------------[3]训练模型------------------------------------------------------------------

try:
    # 重复进行一场场的游戏
    for episode in range(num_episodes):
        obs, info = env.reset()   # 初始化本场游戏的环境
        state = digitize_state(obs)     # 获取初始状态值
        action = np.argmax(q_table[state])      # 根据状态值作出行动决策
        episode_reward = 0
        # 一场游戏分为一个个时间步
        for t in range(max_number_of_steps):  # max_number_of_steps=200
            env.render()    # 更新并渲染游戏画面，注释掉此行可以加快训练速度

            #观测 Observation(Object)：当前step执行后，环境的观测(类型为对象)。例如，从相机获取的像素点，机器人各个关节的角度或棋盘游戏当前的状态等；
            #奖励 Reward(Float): 执行上一步动作(action)后，智能体(agent)获得的奖励(浮点类型)，不同的环境中奖励值变化范围也不相同，但是强化学习的目标就是使得总奖励值最大；
            #完成 Done(Boolen): 表示是否需要将环境重置env.reset。当Done为True时，就表明当前回合(episode)结束。完成200回合或中途实验失败Done为True;
            #信息 Info(Dict): 针对调试过程的诊断信息。在标准的智体仿真评估当中不会使用到这个info，具体用到的时候再说。
            obs, reward, terminated, truncated, info = env.step(action)  # 推进一个时间步长，获取本次行动的反馈结果，返回observation, reward, done, info
            done = terminated or truncated
            # 对致命错误行动进行极大力度的惩罚，让模型恨恨地吸取教训，中途失败直接扣200分。
            if done and t < max_number_of_steps - 1:
                reward = -200

            episode_reward += reward
            action, state = get_action(state, action, obs, reward, episode)  # 作出下一次行动的决策
            if done:
                print('%d Episode finished after %f time steps / mean %f' % (episode+1, t+1, last_time_steps.mean()))
                last_time_steps = np.hstack((last_time_steps[1:], [episode_reward]))  # 更新最近100场游戏的得分stack
                break
                # 如果最近100场平均得分高于195,输出训练成功
            if (last_time_steps.mean() >= goal_average_steps):
                print('Episode %d train agent successfuly!' % episode)
                raise Getoutofloop()
    print('Sorry,Train agent failed!')
except Getoutofloop:
    pass

env.close()
