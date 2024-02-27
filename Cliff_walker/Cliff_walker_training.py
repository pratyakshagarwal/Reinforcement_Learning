import gymnasium as gym
import numpy as np
import pickle as pkl
from Simple_Algorithm.SARSA import SARSA_ALgo
from Simple_Algorithm.q_leraning import q_Learning_ALgo

"""
observation
|3*12 + 1: |37|: possible states|

steps
|0: move up|
|1: move right|
|2: move down|
|3: move left|

rewards
|-1: reward each time step occurs|
|-100: stepping into the cliff|
"""

if __name__ == "__main__":
    # Creates the environment
    cliffEnv = gym.make("CliffWalking-v0")

    # Initializing Q Table
    q_table_input = np.zeros(shape=(48, 4))

    # Parameters
    EPSILON = 0.1
    ALPHA = 0.1
    GAMMA = 0.9
    NUM_EPISODES = 500

    sarsa_cliff_q_table = SARSA_ALgo(q_table_input)
    sarsa_cliff_q_table.train(cliffEnv, num_episode=500, EPSILON=EPSILON, ALPHA=ALPHA, GAMMA=GAMMA)
    sarsa_cliff_q_table.Save_q_table("cliffwalking_sarsa_q_table")

    q_learning_cliff_q_table = q_Learning_ALgo(q_table_input)
    q_learning_cliff_q_table.train(cliffEnv, num_episode=500, EPSILON=EPSILON, ALPHA=ALPHA, GAMMA=GAMMA)
    q_learning_cliff_q_table.Save_q_table("cliffwalking_q_learning_q_table")

