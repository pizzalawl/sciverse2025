import gymnasium as gym
import ale_py
import sys
<<<<<<< HEAD
<<<<<<< HEAD
import stable_baselines3
from stable_baselines3 import PPO
=======
=======
>>>>>>> 4769bb8360c801d744b776c6d1ec430768e8cbfc
from stable_baselines3 import A2C
>>>>>>> 4769bb8360c801d744b776c6d1ec430768e8cbfc

gym.register_envs(ale_py)

env = gym.make("ALE/Tetris-v5", render_mode = "human")

<<<<<<< HEAD
<<<<<<< HEAD
model = PPO("MlpPolicy", env, verbose =1)
model.learn(total_timesteps=10000 , progress_bar=True)
model.save("a2c.tetris")
=======
model = A2C("MlpPolicy", env, verbose =1)
model.learn(total_timesteps=sys.argv[1] , progress_bar=True)
model.save("a2c_tetris.zip")
>>>>>>> 4769bb8360c801d744b776c6d1ec430768e8cbfc
=======
model = A2C("MlpPolicy", env, verbose =1)
model.learn(total_timesteps=sys.argv[1] , progress_bar=True)
model.save("a2c_tetris.zip")
>>>>>>> 4769bb8360c801d744b776c6d1ec430768e8cbfc


