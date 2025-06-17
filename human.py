import gym
import keyboard
import time

# 创建环境（带可视化）
env = gym.make('CartPole-v1', render_mode="human")
obs, info = env.reset()

print("🎮 控制方式：")
print("👉 按住 ← 键：向左")
print("👉 按住 → 键：向右")
print("❌ 按 ESC：退出游戏\n")

try:
    while True:
        env.render()

        # 持续按键检测
        if keyboard.is_pressed("left"):
            action = 0
        elif keyboard.is_pressed("right"):
            action = 1
        else:
            action = 0  # 默认不按时采取某一动作（你也可以设置为随机或保持上一动作）

        # 运行一步环境
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        if done:
            print("💥 游戏结束，重置环境\n")
            time.sleep(1)
            obs, info = env.reset()

        time.sleep(0.05)  # 控制帧率，越小越“丝滑”（0.03~0.05推荐）

except KeyboardInterrupt:
    print("✅ 手动退出。")
finally:
    env.close()
