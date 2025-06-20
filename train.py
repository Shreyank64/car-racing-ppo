import gymnasium as gym
import pybullet_envs_gymnasium  # Required to register PyBullet environments
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize, VecTransposeImage
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
import time

# Environment configuration
env_id = "CarRacing-v3"
num_envs = 8  # Parallel environments for faster training

# Create and wrap the environment
env = make_vec_env(env_id, n_envs=num_envs)
env = VecTransposeImage(env)  # Reorder image channels for CNN input
env = VecNormalize(env, norm_obs=True, norm_reward=True)

# Define PPO model with CNN policy
model = PPO(
    policy="CnnPolicy",
    env=env,
    verbose=1,
    learning_rate=3e-4,
    n_steps=1024,
    batch_size=64,
    n_epochs=10,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.01,
    tensorboard_log="./car_racing_tensorboard/",
    device="cuda",  # Use GPU if available
    policy_kwargs={"normalize_images": False},  # Assume image normalization is handled
)

# Evaluation callback to save the best model
eval_callback = EvalCallback(
    env,
    best_model_save_path="./best_model/",
    log_path="./logs/",
    eval_freq=10_000,
    deterministic=True,
    render=False,
)

# Checkpoint callback to save the model periodically
checkpoint_callback = CheckpointCallback(
    save_freq=20_000,
    save_path="./checkpoints/",
    name_prefix="car_racing_ppo",
)

# Train the model
print("Starting PPO training...")
start_time = time.time()

total_timesteps = 1_000_000
model.learn(total_timesteps=total_timesteps, callback=[eval_callback, checkpoint_callback])

end_time = time.time()

# Save final model and environment statistics
model.save("car_racing_ppo_final2")
env.save("car_racing_env_normalize_final2.pkl")

print(f"Training complete in {round((end_time - start_time) / 3600, 2)} hours. "
      f"Model saved as 'car_racing_ppo_final2'.")
