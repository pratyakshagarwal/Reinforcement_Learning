import os
import gymnasium as gym 
import time
import numpy as np
import tensorflow as tf
from keras.losses import Huber
from Config.Configs import Config
from DP_Algorithm.DQN import DQN_Agent


"""
Observation Space : 

The observation is a ndarray with shape (4,) with the values
 corresponding to the following positions and velocities:
|Num | Observation     | Min   | Max   |
| 0  |Cart Position    |-4.8   | 4.8   |
| 1  |Cart Velocity    |-inf   | inf   |
| 2  |Pole Angle       |~0.418 |~0.418 |
| 3  |Pole Angular Vel |-inf   | inf   |

Action Space :
|0 : Push cart to the left|
|1 : push cart to the right|


Reward :
|+1 :  For every step|
|threshold for rewards is 500 for  v1 and 200 for v0|
 """

CartEnv = gym.make("CartPole-v1")


if __name__ == "__main__":
    # parameters
    obs = 4
    dim1 = 64
    dim2 = 32
    action = 2
    num_episodes = 1000
    batch_size = 64
    loss_fn = Huber(delta=1.0)
    learn_after_steps = 10000
    target_update_after = 4

    config_instance = Config(obs=4, dim1=64, dim2=32, action=2, epochs=1000)
    config_instance.save_configration()
    config_instance.save_to_dataframe_and_csv()

    CartEnv = gym.make('CartPole-v1')
    Cart_dqn = DQN_Agent(obs=obs,
                         dim1=dim1,
                         dim2=dim2,
                         action=action,
                         num_episodes=num_episodes,
                         batch_size=batch_size,
                         loss_fn=loss_fn,
                         learn_after_steps=learn_after_steps,
                         target_update_after=target_update_after)
    
    Cart_dqn.train(CartEnv)
    Cart_dqn.save_q_network('CartPole\Cart_dqn')
    Cart_dqn.save_metric('CartPole\metric_for_cart_dqn')