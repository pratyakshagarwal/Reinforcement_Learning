import gymnasium as gym
import numpy as np
import tensorflow as tf
from keras.losses import MeanSquaredError
from keras.optimizers import Adam 
from Config.Configs import Config
from LunarLander.DQN_Lunar_lander import DQN_LANDER

"""
Observation Space:
| x  | y  |The state is an 8-dimensional vector|
|1, 2|1, 2|The coordinates of the lander in x&y|
|3, 4|3, 4|Its linear velocities in x & y      |
| 5  | 5  |Its angle                           |
| 6  | 6  |its angular velocity                |
|7, 8|7, 8|two booleans that represent whether |
          |each leg is in contact with the     |
          |ground or not.                      |

          
Action Space:
|0 |Do nothing                   |
|1 |Fire left orientation engine |
|2 |Fire main engine             |
|3 |Fire right orientation engine|

  
Rewards:
For each step, the reward:

is increased/decreased the closer/further the lander is to the landing pad.
is increased/decreased the slower/faster the lander is moving.
is decreased the more the lander is tilted (angle not horizontal).
is increased by 10 points for each leg that is in contact with the ground.
is decreased by 0.03 points each frame a side engine is firing.
is decreased by 0.3 points each frame the main engine is firing.

The episode receive an additional reward of -100 or +100 points
 for crashing or landing safely respectively.

Terminate Codn:
An episode is considered a solution if it scores at least 200 points.
"""

if __name__ == "__main__":
    # Enviorment for Lunar Lander
    LLEnv = gym.make('LunarLander-v2')

    # parameters
    obs = 8
    dim1 = 64
    dim2 = 64
    dim3 = 64
    action = 4
    num_episodes = 700
    batch_size =  64
    loss_fn = MeanSquaredError()
    learn_after_steps = 1000
    target_update_after = 4
    optimizer = Adam()

    config_instance = Config(obs=obs, hidden_dims=[dim1, dim2, dim3], action=action, epochs=num_episodes, loss_fn='MeanSquaredError')
    config_instance.save_configration()
    config_instance.save_to_dataframe_and_csv(filename='configuration_for_lunar_lander.csv')

    Mcaragent = DQN_LANDER(obs=obs,
                          dim1=dim1,
                          dim2=dim2,
                          dim3=dim3,
                          action=action,
                          num_episodes=num_episodes,
                          batch_size=batch_size,
                          loss_fn=loss_fn,
                          learn_after_steps=learn_after_steps,
                          target_update_after=target_update_after,
                          optimizer=optimizer)
    
    Mcaragent.train(LLEnv)
    Mcaragent.save_q_network("Lunar_Lander_q_net")
    Mcaragent.save_metric("metric_for_lunar_lander.csv")