import gymnasium  as gym
import numpy as np
import pickle as pkl
import tensorflow as tf
from keras.models import load_model

def policy(state,q_table, explore=0.0):
    action = int(np.argmax(q_table[state]))
    if np.random.random() <= explore:
        action = int(np.random.randint(low=0, high=4, size=1))
    return action

def Evaluate(q_table, Env, NUM_EPISODES):
    for episode in range(NUM_EPISODES):
        done = False
        total_reward = 0
        episode_length = 0

        state = Env.reset()[0]
        Env.render()

        while not done:
            action = policy(state, q_table)
            state, reward, done, _, _ = Env.step(action)

            episode_length+=1
            total_reward+=reward
        print("Episode:{} Episode Length:{} Total Reward:{}".format(episode, episode_length, total_reward))
    Env.close()

class Evaluate_DQN():
    def __init__(self, q_net):
        self.q_net = q_net

    def eval(self, env, num_episode=10):
        for episode in range(num_episode):
            done = False
            state = tf.convert_to_tensor([env.reset()[0]])
            
            while not done:
                env.render()
                action = int(self.policy(state).numpy())
                state, reward, done, _, _ = env.step(action)
                state = tf.convert_to_tensor([state])
        env.close()
        
    def policy(self, state, explore=0.0):
        action = tf.argmax(self.q_net(state)[0], output_type=tf.int32)
        if tf.random.uniform(shape=(), maxval=1) <= explore:
            action = tf.random.uniform(shape=(), minval=0, maxval=8, dtype=tf.int32)
        return action
    

if __name__ == "__main__":
    env = gym.make('CartPole-v1', render_mode='human')
    q_net = load_model("CartPole\Cart_dqn") 
    evaluate = Evaluate_DQN(q_net)
    evaluate.eval(env)

