import gymnasium as gym
import numpy as np
import pickle as pkl
import tensorflow as tf
from keras import Model, Input
from keras.layers import Dense


class SARSA_ALgo():
    def __init__(self, q_table):
        self.q_table = q_table

    def train(self, env, num_episode, EPSILON=0.0, ALPHA=0.0, GAMMA=0.9):
        for episode in range(num_episode):
            # Intializing episode
            done = False
            total_reward = 0
            episode_length = 0

            # set intial state
            state = env.reset()[0]

            # selecting an action according to out policy
            action = self.policy(state, EPSILON)

            while not done:
                # Take an action in the Enviorment
                next_state, reward, done, _, _   = env.step(action)

                # select the next action
                next_action = self.policy(next_state, EPSILON)

                # sarsa update
                self.q_table[state][action] += ALPHA* (reward + GAMMA * self.q_table[next_state][next_action] - self.q_table[state][action])

                state = next_state
                action = next_action

                total_reward += reward
                episode_length += 1

            print("Episode:{} Episode Length:{} Total Reward:{}".format(episode, episode_length, total_reward))
        env.close()

    def policy(self, state, explore=0.0):
        action = int(np.argmax(self.q_table[state]))
        if np.random.random() <= explore:
            action = int(np.random.randint(low=0, high=4, size=1))

        return action
    
    def Save_q_table(self, filename):
        pkl.dump(self.q_table, open(filename, 'wb'))
        print("saved {}".format(filename))