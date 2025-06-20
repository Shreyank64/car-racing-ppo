# 🏎️ CarRacing-v3 PPO Agent

This project trains a reinforcement learning (RL) agent to drive in the `CarRacing-v3` environment from [Gymnasium](https://gymnasium.farama.org/) using **Proximal Policy Optimization (PPO)** from [Stable-Baselines3](https://stable-baselines3.readthedocs.io/).

The agent learns directly from pixel inputs using a convolutional neural network (CNN) policy and is optimized with vectorized environments and reward normalization for faster convergence.

---

## 📦 Features

- ✅ Vectorized parallel environments for faster training
- ✅ PPO algorithm with CNN-based policy
- ✅ `VecNormalize` for stable learning and reward scaling
- ✅ Automatic evaluation and checkpointing during training
- ✅ TensorBoard support for visualizing training progress

---

## 📂 Project Structure
car-racing-ppo/
├── train.py # Training script
├── test.py # Inference script
├── requirements.txt # Python dependencies
├── README.md # Project documentation
├── car_racing_ppo_final2.zip # (Optional) Final saved model
├── car_racing_env_normalize_final2.pkl # Normalization statistics
├── logs/ # TensorBoard logs
├── checkpoints/ # Model checkpoints
└── best_model/ # Best-performing model
