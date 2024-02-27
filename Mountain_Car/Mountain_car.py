import gymnasium as gym
import numpy as np
import pandas as pd
from keras.losses import Huber
from keras.optimizers import Adam 
from Config.Configs import Config
from Mountain_Car.DQN_Mountain_car import DQN_CAR

"""
Observation space:
|0 | postion of the car along the x-axis|
|1 | velocity of car                    |

Action space:
|0 |Accelerate to the left |
|1 |Don't accelerate       |
|2 |Accelerate to the right|

Reward:
The goal is to reach the flag placed on top of the right hill as quickly as possible,
 as such the agent is penalised with a reward of -1 for each timestep.
"""

if __name__ == "__main__":
    # Enviorment for mountain car
    McarEnv = gym.make('MountainCar-v0')

    # parameters
    obs = 2
    dim1 = 64
    dim2 = 32
    action = 3
    num_episodes = 1000
    batch_size =  64
    loss_fn = Huber(delta=1.0)
    learn_after_steps = 1000
    target_update_after = 4
    optimizer = Adam(lr=0.001)

    config_instance = Config(obs=2, dim1=64, dim2=32, action=3, epochs=1000)
    config_instance.save_configration()
    config_instance.save_to_dataframe_and_csv(filename='configuration_for_mountain_car.csv')

    Mcaragent = DQN_CAR(obs=obs,
                          dim1=dim1,
                          dim2=dim2,
                          action=action,
                          num_episodes=num_episodes,
                          batch_size=batch_size,
                          loss_fn=loss_fn,
                          learn_after_steps=learn_after_steps,
                          target_update_after=target_update_after,
                          optimizer=optimizer)
    
    Mcaragent.train(McarEnv)
    Mcaragent.save_q_network("mountain_car_q_net")
    Mcaragent.save_metric("metric_for_mountain_car.csv")

