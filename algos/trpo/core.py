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

def hessian_vector_product(x, f, module, retain=True):
    gradients = torch.autograd.grad(f, module.parameters(), create_graph=True)
    flat_gradients = params_flat(gradients)
    flat_gradients = (flat_gradients * x).sum()
    hessian = torch.autograd.grad(
                flat_gradients, module.parameters(),
                allow_unused=True, retain_graph=retain)
    return params_flat(hessian)

def params_flat(params):
    flat_params = []
    for p in params:
        flat_params.append(p.reshape(-1))
    return torch.cat(flat_params)

class discreate_policy(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_sizes):
        super().__init__()
        layer = [nn.Linear(obs_dim, hidden_sizes[0]), nn.ReLU()]
        for i in range(1, len(hidden_sizes)):
            layer.append(nn.Linear(hidden_sizes[i-1], hidden_sizes[i]))
            layer.append(nn.ReLU())
        layer.append(nn.Linear(hidden_sizes[-1], act_dim))
        self.main = nn.Sequential(*layer)

    def forward(self, obs):
        return self.main(obs)

    def kl(self, logp0, logp1):
        return (logp1.exp() * (logp1 - logp0)).sum(1).mean()

class value_function(nn.Module):
    def __init__(self, obs_dim, hidden_sizes):
        super().__init__()
        layer = [nn.Linear(obs_dim, hidden_sizes[0]), nn.ReLU()]
        for i in range(1, len(hidden_sizes)):
            layer.append(nn.Linear(hidden_sizes[i-1], hidden_sizes[i]))
            layer.append(nn.ReLU())
        layer.append(nn.Linear(hidden_sizes[-1], 1))
        self.main = nn.Sequential(*layer)

    def forward(self, obs):
        return self.main(obs)

class actor_critic(nn.Module):
    def __init__(self, act_dim, obs_dim, hidden_sizes):
        super().__init__()
        self.value_function = value_function(obs_dim, hidden_sizes)
        self.policy = discreate_policy(obs_dim, act_dim, hidden_sizes)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

        self.old_policy = discreate_policy(obs_dim, act_dim, hidden_sizes)
        self.copy_policy_param()

    def copy_policy_param(self):
        self.old_policy.load_state_dict(self.policy.state_dict())

    def policy_update(self, param):
        start_idx = 0
        for p in self.policy.parameters():
            n = p.nelement()
            p.data = param[start_idx:start_idx + n].reshape_as(p)
            start_idx += n

    def sampling_action(self, obs):
        logits = self.policy(obs)
        return torch.multinomial(logits.softmax(1), 1).squeeze()

    def likelihood(self, obs, act):
        logits = self.policy(obs)
        logp_all = F.log_softmax(logits, 1)
        logp = logp_all * torch.eye(logp_all.shape[-1])[act.long()]
        return logp_all, logp.sum(1)

    def old_likelihood(self, obs, act):
        logits = self.old_policy(obs)
        logp_all = F.log_softmax(logits, 1)
        logp = logp_all * torch.eye(logp_all.shape[-1])[act.long()]
        return logp_all, logp.sum(1)