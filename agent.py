import torch
from torch.distributions import Categorical
from torch.optim import Adam
from torch.utils.data import DataLoader
from actor_critic_network import ActorCriticNetwork
from batch_dataset import Batch_DataSet
import os


class Agent():
    def __init__(self, in_channels, n_output, learning_rate, gamma, n_updates, clip, minibatch_size, c1, c2):
        self.actor_critic = ActorCriticNetwork(in_channels, n_output)
        
        self.gamma = gamma
        self.n_updates_per_iteration = n_updates
        self.clip = clip
        self.minibatch_size = minibatch_size
        self.c1 = c1
        self.c2 = c2
        self.model_name = 'modelo_actor_critico.pt'

        self.actor_critic_optimizer = Adam(self.actor_critic.parameters(), lr=learning_rate)

        self.load_models()

    def save_models(self):
        torch.save(self.actor_critic.state_dict(), self.model_name)
    
    def load_models(self):
        if(os.path.isfile(self.model_name)):
            print('Se ha cargado un modelo para la red neuronal')
            self.actor_critic.load_state_dict(torch.load(self.model_name))
        else:
            print('No se ha encontrado ningun podelo para la red neuronal')
    
    def get_action(self, observation):
        distribution, value = self.actor_critic(observation)
        m = Categorical(distribution.squeeze(0))
        action = m.sample()
        log_prob = m.log_prob(action)

        return log_prob, value.squeeze(0).squeeze(0), action

    def get_action_max_prob(self, observation):
        distribution, _ = self.actor_critic(observation)
        action = torch.argmax(distribution)

        return action

    def get_log_probs_batch(self, observations, actions):
        distributions, values = self.actor_critic(observations)
        m = Categorical(distributions)
        log_probs = m.log_prob(actions)     
        entropy = m.entropy()

        return log_probs, values, entropy

    def update(self, observations, actions, advantage_values, old_logprobs):
        print('Actualizando...')
        
        dataset = Batch_DataSet(observations, actions, advantage_values, old_logprobs)
        dataloader = DataLoader(dataset, batch_size=self.minibatch_size, num_workers=0, shuffle=True)
        
        for _ in range(self.n_updates_per_iteration):

            for i, batch in enumerate(dataloader):

                if i > 8:
                    break

                observations_batch, actions_batch, advantages_batch, old_action_prob_batch = batch 

                advantages_batch = (advantages_batch - torch.mean(advantages_batch) ) / ( torch.std(advantages_batch) + 1e-8)

                current_log_probs, current_values, entropy = self.get_log_probs_batch(observations_batch, actions_batch)

                current_values = current_values.squeeze(1)

                # Si vas a utilizar el LOGARITMO de las probabilidades entonces el ratio se calcula con el exponente
                # Pero si vas a dividir las probabilidades entonces debes usar las probabilidades a secas sin calcular su logaritmo
                # current_probs / old_probs == torch.export(torch.log(current_probs) - torch.log(old_probs))

                ratios = torch.exp(current_log_probs - old_action_prob_batch)

                surr1 = ratios * advantages_batch
                surr2 = torch.clamp(ratios, 1-self.clip, 1+self.clip) * advantages_batch
                actor_loss = -torch.min(surr1,surr2)

                critic_loss = torch.pow(advantages_batch - current_values,2)

                ac_loss = actor_loss.mean() + self.c1 * critic_loss.mean() - self.c2 * entropy.mean()

                self.actor_critic_optimizer.zero_grad()
                ac_loss.backward()
                self.actor_critic_optimizer.step()


