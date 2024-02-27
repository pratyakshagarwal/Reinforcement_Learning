import gymnasium  as gym
import numpy as np
import pickle as pkl
import tensorflow as tf
from keras.models import load_model
from Evalutator.Evaluator import Evaluate_DQN

if __name__ == "__main__":
    env = gym.make('LunarLander-v2', render_mode='human')
    q_net = load_model("Lunar_Lander_q_net") 
    evaluate = Evaluate_DQN(q_net)
    evaluate.eval(env)