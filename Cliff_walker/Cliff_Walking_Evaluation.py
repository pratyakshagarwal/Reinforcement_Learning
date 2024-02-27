import gymnasium as gym
import numpy as np
import pickle as pkl
from Evalutator.Evaluator import Evaluate

cliffEnv = gym.make('CliffWalking-v0', render_mode='human')

sarsa_q_table = pkl.load(open("Cliff_walker\cliffwalking_sarsa_q_table", "rb"))
q_learning_q_table = pkl.load(open("Cliff_walker\cliffwalking_q_learning_q_table", "rb"))


NUM_EPISODES = 5

# Evaluate(q_table=sarsa_q_table, Env=cliffEnv, NUM_EPISODES=NUM_EPISODES)
Evaluate(q_table=q_learning_q_table, Env=cliffEnv, NUM_EPISODES=NUM_EPISODES)