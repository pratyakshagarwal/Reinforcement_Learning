import gymnasium as gym
import numpy as np
import pickle as pkl
import tensorflow as tf
from keras import Model, Input
from keras.layers import Dense

class Q_Net(tf.keras.models.Model):
    def __init__(self, obs, dim1, dim2, action, loss, optimizer=None):
        super().__init__()

        self.q_net = tf.keras.Sequential([
            Dense(dim1, input_dim=obs, activation='relu'),
            Dense(dim2, activation='relu'),
            Dense(action, activation='linear')
        ])

        if optimizer:
            self.optimizer = optimizer  # Set the optimizer attribute
            self.model.compile(optimizer=self.optimizer, loss=loss)
        else:
            self.optimizer = tf.keras.optimizers.Adam()  # Default optimizer if none provided
            self.q_net.compile(optimizer=self.optimizer, loss=loss)

    def call(self, state):
        # Add a batch dimension of 1 to the state
        state = tf.expand_dims(state, axis=0)
        return self.q_net(state)
        
    def get_summary(self):
        return self.q_net.summary()



class Q_Learning_ALGO_DP(tf.keras.layers.Layer):
    def __init__(self, obs, dim1, dim2, action, loss, optimizer=None):
        self.q_net = Q_Net(obs, dim1, dim2, action, loss, optimizer)

    def train(self, env, num_episode, epsilon=1.0, epsilon_decay=1.001, gamma=0.99, alpha=0.001):
        for episode in range(num_episode):
            
            # initialize
            done = False
            total_reward = 0
            episode_length = 0

            state = tf.convert_to_tensor([env.reset()[0]])

            while not done:
                action = self.policy(state, epsilon)
                next_state, reward, done, _ , _ = env.step(action.numpy())
                next_state = tf.convert_to_tensor([next_state])
                next_action = self.policy(next_state, epsilon)
                
                target = reward + gamma * self.q_net(next_state)[0][next_action]
                if done:
                    target = reward

                with tf.GradientTape() as tape:
                    current = self.q_Net(state)

                grads = tape.gradient(current, self.q_net.trainable_weights)
                delta = target - current[0][action]
                for j in range(len(grads)):
                    self.q_net.trainable_weights[j].assign_add(alpha * delta * grads[j])

                state - next_state

                total_reward += reward
                episode_length+=1

            print("Episode:", episode, "Length:", episode_length, "Rewards:", total_reward, "Epsilon:", epsilon)
            epsilon /= epsilon_decay
    
    def policy(self, state, explore):
        action = tf.argmax(self.q_net(state)[0], output_type=tf.int32)
        if tf.random.uniform(shape=(), maxval=1) <= explore:
            action = tf.random.uniform(shape=(), minval=0, maxval=2, dtype=tf.int32)
        return action
    
    def save_q_net(self, filename):
        self.q_net.save(filename)
        print("q_net_saved")