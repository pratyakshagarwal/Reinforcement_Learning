import os
import gym 
import time
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy

env = gym.make("FrozenLake-v1", render_mode='human')
env.reset()
env.render()
print('Intial state of the system')

numberofIteration = 50

for i in range(numberofIteration):
    randomAction = env.action_space.sample()
    returnValue = env.step(randomAction)
    env.render()
    print('Iteration: {} and action {}'.format(i+1,randomAction))
    time.sleep(2)
    if returnValue[2]:
        break

env.close()
log_path = os.path.join('Training', 'Logs')

env = DummyVecEnv([lambda: env])
model  = PPO('MlpPolicy', env, verbose = 1, tensorboard_log=log_path)

model.learn(total_timesteps=1000)