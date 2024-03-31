import gymnasium as gym
from RL_Algorithms.Actor_Critic.a2c import Agent

if __name__ == '__main__':
    env_name = "CartPole-v1"
    env = gym.make(env_name)

    # parameters
    n_observation = env.observation_space.shape
    n_action = env.action_space.n 

    # hyperparamters
    episodes = 600
    batch_size = 128
    hidden_dim = [256, 256]
    name = 'cartpolea2c'
    dir = 'crtpolemodel'

    agent = Agent(env_name, n_action, n_observation, hidden_dim=hidden_dim)
    # agent.run(episodes, dir=dir, name=name)
    agent.evaluate(5, dir=dir, name=r'600cartpolea2c.weights.h5')