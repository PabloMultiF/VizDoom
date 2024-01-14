from agent import Agent
from frames_env import FramesEnv
from vizdoomenv import Vizdoomenv
from worker import Worker
import torch
import numpy as np
from auxiliars import compute_gae

def train(arguments):
    learning_rate = arguments.lr
    gamma = arguments.gamma
    n_updates = arguments.epochs
    clip = arguments.clip
    c1 = arguments.c1
    c2 = arguments.c2
    minibatch_size = arguments.minibatch_size
    batch_size = 129
    cicles = 1000
    lam = 0.95
    in_channels = 1
    #in_channels = 4
    n_outputs = 3
    actors = 8
    #actors = 1

    agente = Agent(in_channels, n_outputs, learning_rate, gamma, n_updates, clip, minibatch_size, c1, c2)

    env_runners = []
    for _ in range(actors):
        raw_env = Vizdoomenv()
        env = FramesEnv(raw_env)
        env_runners.append(Worker(env, agente, batch_size))

    for _ in range(cicles):
        batch_observations, batch_actions, batch_advantages, batch_old_action_prob = None, None, None, None
        
        for env_runner in env_runners:
            obs, actions, rewards, dones, values, old_action_prob = env_runner.run()

            advantages = compute_gae(rewards, values, dones, lam)
            
            batch_observations = torch.stack(obs[:-1]) if batch_observations == None else torch.cat([batch_observations,torch.stack(obs[:-1])])
            batch_actions = np.stack(actions[:-1]) if batch_actions is None else np.concatenate([batch_actions,np.stack(actions[:-1])])
            batch_advantages = advantages if batch_advantages is None else np.concatenate([batch_advantages,advantages])
            batch_old_action_prob = torch.stack(old_action_prob[:-1]) if batch_old_action_prob == None else torch.cat([batch_old_action_prob,torch.stack(old_action_prob[:-1])])

        agente.update(batch_observations, batch_actions, batch_advantages, batch_old_action_prob)
        agente.save_models()


if __name__ == '__main__':
    train()


















