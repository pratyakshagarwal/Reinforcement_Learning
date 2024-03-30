import gymnasium as gym
from RL_Algorithms.Q_Learning import Q_Learning

if __name__ == '__main__':
    env_name = 'MountainCar-v0'
    env = gym.make(env_name)

    # hyper paramters
    episodes = 22500
    learning_rate = 0.1
    epsilon_decay = 1.0003
    name = 'cartpole'

    # enviorment parameters
    n_observation = env.observation_space.shape
    n_action = env.action_space.n

    agent = Q_Learning(env_name, n_observation=n_observation, n_action=n_action)
    # agent.run(episodes=episodes, learning_rate=learning_rate, epsilon_decay=epsilon_decay, name=name)
    agent.evaluate(10, name=r'qtables\22000cartpole-qtable.npy')