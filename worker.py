import torch

class Worker():
    def __init__(self, env, agente, batch_size):
        self.env = env
        self.agente = agente
        self.observation = self.env.reset()
        self.batch_size = batch_size
        
    def run(self):
        observations = []
        actions = []
        rewards = []
        dones = []
        values = []
        log_probs = []
        
        for _ in range(self.batch_size):
            self.observation = torch.FloatTensor(self.observation).unsqueeze(0)

            log_prob, value, action = self.agente.get_action(self.observation)

            next_observation, reward, done = self.env.step(action.item())

            if done:
                next_observation = self.env.reset()

            observations.append(self.observation.squeeze(0))
            actions.append(action)
            values.append(value.detach())
            log_probs.append(log_prob.detach())
            rewards.append(reward)
            dones.append(done)

            self.observation = next_observation
        
        return [observations, actions, rewards, dones, values, log_probs]
