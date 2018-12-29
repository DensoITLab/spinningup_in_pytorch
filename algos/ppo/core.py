import torch
import torch.nn as nn
import torch.nn.functional as F

import scipy.signal

def discount_cumsum(x, discount):
    """
    magic from rllab for computing discounted cumulative sums of vectors.

    input: 
        vector x, 
        [x0, 
         x1, 
         x2]

    output:
        [x0 + discount * x1 + discount^2 * x2,  
         x1 + discount * x2,
         x2]
    """
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]

def onehot(idx, size):
    return torch.eye(size)[idx.long()]

class discreate_policy(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_size):
        super().__init__()
        layer = [nn.Linear(obs_dim, hidden_size[0]), nn.Tanh()]
        for i in range(1,len(hidden_size)):
            layer += [
                nn.Linear(hidden_size[i-1], hidden_size[i]),
                nn.Tanh()
            ]
        layer.append(nn.Linear(hidden_size[-1], act_dim))
        self.main = nn.Sequential(*layer)

    def forward(self, obs):
        return self.main(obs)

class value_function(nn.Module):
    def __init__(self, obs_dim, hidden_size):
        super().__init__()
        layer = [nn.Linear(obs_dim, hidden_size[0]), nn.Tanh()]
        for i in range(1,len(hidden_size)):
            layer += [
                nn.Linear(hidden_size[i-1], hidden_size[i]),
                nn.Tanh()
            ]
        layer.append(nn.Linear(hidden_size[-1], 1))
        self.main = nn.Sequential(*layer)

    def forward(self, obs):
        return self.main(obs).squeeze()

class actor_critic(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_size):
        super().__init__()
        self.value_function = value_function(obs_dim, hidden_size)
        self.policy = discreate_policy(obs_dim, act_dim, hidden_size)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

    def sampling_action(self, obs, with_likelihood=True):
        logits = self.policy(obs)
        action = torch.multinomial(logits.softmax(1), 1)
        if with_likelihood:
            logp_all = F.log_softmax(logits, 1)
            logp = (onehot(action[0], logp_all.shape[1]) * logp_all[0]).sum()
            return action.squeeze().item(), logp.squeeze().item()
        return action.squeeze().item()

    def likelihood(self, obs, act):
        logits = self.policy(obs)
        logp_all = F.log_softmax(logits, 1)
        pi = torch.multinomial(logits.softmax(1), 1).squeeze()
        logp = (onehot(act, logp_all.shape[1]) * logp_all).sum(1)
        return logp
