import gymnasium as gym
from Evalutator.Evaluator import Evaluate_DQN
from keras.models import load_model

if __name__ == "__main__":
    env = gym.make('CartPole-v1', render_mode='human')
    q_net = load_model("CartPole\Cart_dqn") 
    evaluate = Evaluate_DQN(q_net)
    evaluate.eval(env)