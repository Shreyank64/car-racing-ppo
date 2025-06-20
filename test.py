import gymnasium as gym
import pybullet_envs_gymnasium  # Required to register the environment
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

# Load the CarRacing-v3 environment with rendering enabled
env_id = "CarRacing-v3"
env = gym.make(env_id, render_mode="human", max_episode_steps=5000)

# Wrap in DummyVecEnv for compatibility with VecNormalize
env = DummyVecEnv([lambda: env])

# Load normalization statistics and trained model
env = VecNormalize.load("car_racing_env_normalize_final2.pkl", env)
env.training = False  # Disable reward normalization
env.norm_reward = False

model = PPO.load("car_racing_ppo_final2", device="cuda")

# Run one test episode
obs = env.reset()
done = False
steps = 0

while not done:
    action, _ = model.predict(obs)
    obs, reward, done, info = env.step(action)
    env.render()
    steps += 1

env.close()
print(f"Test episode completed in {steps} steps.")
