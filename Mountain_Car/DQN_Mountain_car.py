# ========== Deep Q-Learning Implementation =========
import gymnasium as gym
import pandas as pd
import numpy as np
import tensorflow as tf
from keras.models import clone_model
from keras import Model, Input
from keras.layers import Dense
from keras.losses import Huber
from keras import Sequential

class Q_Net(tf.keras.models.Model):
    def __init__(self, obs, dim1, dim2, action, loss, optimizer=None):
        super().__init__()

        self.model = tf.keras.Sequential([
            Dense(dim1, input_dim=obs, activation='relu'),
            Dense(dim2, activation='relu'),
            Dense(action, activation='linear')
        ])

        if optimizer:
            self.optimizer = optimizer  # Set the optimizer attribute
            self.model.compile(optimizer=self.optimizer, loss=loss)
        else:
            self.optimizer = tf.keras.optimizers.Adam()  # Default optimizer if none provided
            self.model.compile(optimizer=self.optimizer, loss=loss)

    def call(self, state):
        return self.model(state)
        
    def get_summary(self):
        return self.model.summary()


class ReplayBuffer:
    def __init__(self, max_transition):
        super(ReplayBuffer, self).__init__()
        self.max_transitions = max_transition
        self.buffer = []

    def insert_transition(self, transtion):
        if len(self.buffer) >= self.max_transitions:
            self.buffer.pop(0)
        self.buffer.append(transtion)

    def sample_transition(self, batch_size=16):
        random_indices = tf.random.uniform(shape=(batch_size, ), minval=0, maxval=len(self.buffer), dtype=tf.int32)
        sampled_current_states, sampled_actions, sampled_rewards, sampled_next_states, sampled_terminals = zip(*[self.buffer[index] for index in random_indices])

        return (
            tf.convert_to_tensor(sampled_current_states),
            tf.convert_to_tensor(sampled_actions),
            tf.convert_to_tensor(sampled_rewards),
            tf.convert_to_tensor(sampled_next_states),
            tf.convert_to_tensor(sampled_terminals),
        )



class DQN_CAR:
    def __init__(self,obs, dim1, dim2, action, num_episodes, batch_size, loss_fn, learn_after_steps, target_update_after,
                 epsilon=1.0, epsilon_decay=1.005, gamma=0.99, max_transitions=100000, optimizer=None, verbose=1):
        super(DQN_CAR, self).__init__()
        self.q_net = Q_Net(obs=obs, dim1=dim1, dim2=dim2, action=action, loss='mse', optimizer=optimizer)
        self.target_net = Q_Net(obs=obs, dim1=dim1, dim2=dim2, action=action, loss='mse', optimizer=optimizer)
        self.replay_buffer = ReplayBuffer(max_transition=max_transitions)
        self.num_episodes = num_episodes
        self.batch_size = batch_size
        self.loss_fn = loss_fn
        self.learn_after_steps = learn_after_steps
        self.target_update_after = target_update_after
        self.step_counter = 0
        self.metric = {"episode": [], "length": [], "total_reward": [], "exploration": []}
        # self.random_states = Metric(self.q_net).gather_random_states(num_states=20)
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.max_transitions = max_transitions
        self.verbose = verbose

    def train(self, env):
        for episode in range(self.num_episodes):
            done = False
            total_reward  = 0
            episode_length = 0
            state = env.reset()[0]

            while not done:
                action = self.policy(state, explore=self.epsilon)
                next_state, reward, done, _, _ = env.step(action.numpy())
                self.replay_buffer.insert_transition([state, action, reward, next_state, done])
                state = next_state
                self.step_counter += 1

                if self.step_counter % self.learn_after_steps == 0:
                    current_states, actions, rewards, next_states, terminals  = self.replay_buffer.sample_transition(self.batch_size)
                    # next_actions_values = tf.reduce_max(self.target_net(next_state), axis=1)
                    next_actions_values = tf.reduce_max(self.target_net(tf.expand_dims(next_state, axis=0)), axis=1)
                    targets = tf.where(terminals, rewards, rewards+self.gamma*next_actions_values)

                    with tf.GradientTape() as tape:
                        preds = self.q_net(current_states)
                        batch_nums = tf.range(0, limit=self.batch_size)
                        indices = tf.stack((batch_nums, actions), axis=1)
                        current_values = tf.gather_nd(preds, indices)
                        loss = self.loss_fn(targets, current_values)
                    grads = tape.gradient(loss, self.q_net.trainable_weights)
                    self.q_net.optimizer.apply_gradients(zip(grads, self.q_net.trainable_weights))

                if self.step_counter % self.target_update_after == 0:
                    self.target_net.set_weights(self.q_net.get_weights())

                total_reward += reward
                episode_length += 1

                # avg_q = tf.reduce_mean(Metric(self.q_net).get_q_values(self.random_states)).numpy()
            self.metric["episode"].append(episode)
            self.metric["length"].append(episode_length)
            self.metric["total_reward"].append(total_reward)
            # self.metric["avg_q"].append(avg_q)
            self.metric["exploration"].append(self.epsilon)
            self.epsilon /= self.epsilon_decay
            if self.verbose == 1:
                print("Episode:{} Episode Length:{} Total Reward:{} EPSILON: {}".format(episode, episode_length, total_reward, self.epsilon))

        env.close()

    # def calculate_reward(self, state):
    #     reward = -1.0
    #     reward = -1.0
    #     if -0.5 <= state[0] <= 0.5 and -1 <= state[1] <= 1 and -0.07 <= state[2] <= 0.07 and -0.525 <= state[3] <= 0.525:
    #         reward = 1.0
    #     return reward
    
    def policy(self, state, explore=0.0):
        action = tf.argmax(self.q_net(tf.expand_dims(state, axis=0))[0], output_type=tf.int32)
        if tf.random.uniform(shape=(), maxval=1) <= explore:
            action = tf.random.uniform(shape=(), maxval=2, dtype=tf.int32)
        return action
    
    def save_q_network(self, filename):
        """
        Save the Q-network to a file.

        Args:
            filename (str): Name of the file to save the Q-network.
        """
        self.q_net.save(filename)

    def save_metric(self, filename):
        """
        Save the metric data to a file.

        Args:
            filename (str): Name of the file to save the metric data.
        """
        pd.DataFrame(self.metric).to_csv(filename, index=False)