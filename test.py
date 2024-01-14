import torch
from agent import Agent
from frames_env import FramesEnv
from vizdoomenv import Vizdoomenv

def test(arguments):
    learning_rate = arguments.lr
    gamma = arguments.gamma
    n_updates = arguments.epochs
    clip = arguments.clip
    c1 = arguments.c1
    c2 = arguments.c2
    minibatch_size = arguments.minibatch_size
    in_channels = 1
    n_outputs = 3

    agente = Agent(in_channels, n_outputs, learning_rate, gamma, n_updates, clip, minibatch_size, c1, c2)

    raw_env = Vizdoomenv()
    env = FramesEnv(raw_env)

    observation = env.reset()

    while True:
        observation = torch.FloatTensor(observation).unsqueeze(0)
                    
        action = agente.get_action_max_prob(observation)

        observation, _, done = env.step(action.item())

        if done:
            observation = env.reset()

if __name__ == '__main__':
    test()


















