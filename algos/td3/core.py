import torch
import torch.nn as nn

import copy

class continuous_policy(nn.Module):
    def __init__(self, act_dim, obs_dim, hidden_layer=(400,300)):
        super().__init__()
        layer = [nn.Linear(obs_dim, hidden_layer[0]), nn.ReLU()]
        for i in range(1, len(hidden_layer)):
            layer.append(nn.Linear(hidden_layer[i-1], hidden_layer[i]))
            layer.append(nn.ReLU())
        layer.append(nn.Linear(hidden_layer[-1], act_dim))
        layer.append(nn.Tanh())
        self.policy = nn.Sequential(*layer)

    def forward(self, obs):
        return self.policy(obs)

class q_function(nn.Module):
    def __init__(self, obs_dim, hidden_layer=(400,300)):
        super().__init__()
        layer = [nn.Linear(obs_dim, hidden_layer[0]), nn.ReLU()]
        for i in range(1, len(hidden_layer)):
            layer.append(nn.Linear(hidden_layer[i-1], hidden_layer[i]))
            layer.append(nn.ReLU())
        layer.append(nn.Linear(hidden_layer[-1], 1))
        self.policy = nn.Sequential(*layer)

    def forward(self, obs):
        return self.policy(obs)

class actor_critic(nn.Module):
    def __init__(self, act_dim, obs_dim, hidden_layer=(400,300), act_limit=2):
        super().__init__()
        self.policy = continuous_policy(act_dim, obs_dim, hidden_layer)

        self.q1 = q_function(obs_dim+act_dim, hidden_layer)
        self.q2 = q_function(obs_dim+act_dim, hidden_layer)
        self.act_limit = act_limit

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

        self.policy_targ = continuous_policy(act_dim, obs_dim, hidden_layer)
        self.q1_targ = q_function(obs_dim+act_dim, hidden_layer)
        self.q2_targ = q_function(obs_dim+act_dim, hidden_layer)

        self.copy_param()

    def copy_param(self):
        self.policy_targ.load_state_dict(self.policy.state_dict())
        self.q1_targ.load_state_dict(self.q1.state_dict())
        self.q2_targ.load_state_dict(self.q2.state_dict())
        # for m_targ, m_main in zip(self.policy_targ.modules(), self.policy.modules()):
        #     if isinstance(m_targ, nn.Linear):
        #         m_targ.weight.data = m_main.weight.data
        #         m_targ.bias.data = m_main.bias.data

        # for m_targ, m_main in zip(self.q_targ.modules(), self.q.modules()):
        #     if isinstance(m_targ, nn.Linear):
        #         m_targ.weight.data = m_main.weight.data
        #         m_targ.bias.data = m_main.bias.data

    def get_action(self, obs, noise_scale):
        pi = self.act_limit * self.policy(obs)
        pi += noise_scale * torch.randn_like(pi)
        pi.clamp_(max=self.act_limit, min=-self.act_limit)
        return pi.squeeze()

    def get_target_action(self, obs, noise_scale, clip_param):
        pi = self.act_limit * self.policy_targ(obs)
        eps = noise_scale * torch.randn_like(pi)
        eps.clamp_(max=clip_param, min=-clip_param)
        pi += eps
        pi.clamp_(max=self.act_limit, min=-self.act_limit)
        return pi.detach()

    def update_target(self, rho):
        # compute rho * targ_p + (1 - rho) * main_p
        for poly_p, poly_targ_p in zip(self.policy.parameters(), self.policy_targ.parameters()):
            poly_targ_p.data = rho * poly_targ_p.data + (1-rho) * poly_p.data

        for q_p, q_targ_p in zip(self.q1.parameters(), self.q1_targ.parameters()):
            q_targ_p.data = rho * q_targ_p.data + (1-rho) * q_p.data

        for q_p, q_targ_p in zip(self.q2.parameters(), self.q2_targ.parameters()):
            q_targ_p.data = rho * q_targ_p.data + (1-rho) * q_p.data


    def compute_target(self, obs, pi, gamma, rewards, done):
        # compute r + gamma * (1 - d) * Q(s', mu_targ(s'))
        q1 = self.q1_targ(torch.cat([obs, pi], -1))
        q2 = self.q2_targ(torch.cat([obs, pi], -1))
        q  = torch.min(q1, q2)
        return (rewards + gamma * (1-done) * q.squeeze()).detach()

    def q_function(self, obs, detach=True, action=None):
        # compute Q(s, a) or Q(s, mu(s))
        if action is None:
            pi = self.act_limit * self.policy(obs)
        else:
            pi = action
        if detach:
            pi = pi.detach()
        return self.q1(torch.cat([obs, pi], -1)).squeeze(), self.q2(torch.cat([obs, pi], -1)).squeeze()