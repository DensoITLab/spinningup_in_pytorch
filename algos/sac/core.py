import torch
import torch.nn as nn

import math

EPS = 1e-8
LOG_STD_MAX = 2
LOG_STD_MIN = -20

class gaussian_policy(nn.Module):
    def __init__(self, act_dim, obs_dim, hidden_layer=(400,300)):
        super().__init__()
        layer = [nn.Linear(obs_dim, hidden_layer[0]), nn.ReLU()]
        for i in range(1, len(hidden_layer)):
            layer.append(nn.Linear(hidden_layer[i-1], hidden_layer[i]))
            layer.append(nn.ReLU())
        self.main = nn.Sequential(*layer)
        self.mu   = nn.Linear(hidden_layer[-1], act_dim)
        self.log_std = nn.Sequential(
            nn.Linear(hidden_layer[-1], act_dim),
            nn.Tanh()
        )

    def forward(self, obs):
        f = self.main(obs)
        mu, log_std = self.mu(f), self.log_std(f)
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)
        
        pi = mu + torch.randn_like(mu) * log_std.exp()
        logp_pi = self.gaussian_likelihood(pi, mu, log_std)
        return mu, pi, logp_pi

    def gaussian_likelihood(self, pi, mu, log_std):
        return  torch.sum(-0.5 * (((pi-mu)/(log_std.exp()+EPS))**2 + 2 * log_std + math.log(2*math.pi)), 1)

class value_function(nn.Module):
    def __init__(self, inp_dim, hidden_layer=(400,300)):
        super().__init__()
        layer = [nn.Linear(inp_dim, hidden_layer[0]), nn.ReLU()]
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
        self.policy = gaussian_policy(act_dim, obs_dim, hidden_layer)

        self.q1 = value_function(obs_dim+act_dim, hidden_layer)
        self.q2 = value_function(obs_dim+act_dim, hidden_layer)
        self.v  = value_function(obs_dim, hidden_layer)
        self.act_limit = act_limit

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

        self.v_targ = value_function(obs_dim, hidden_layer)

        self.copy_param()

    def pass_gradient_clip(self, x, low, high):
        clip_high = (x > high).float()
        clip_low  = (x < low).float()
        return x + ((high - x) * clip_high + (low - x) * clip_low).detach()

    def squashing(self, mu, pi, logp_pi):
        mu = mu.tanh()
        pi = pi.tanh()
        logp_pi = logp_pi - (self.pass_gradient_clip(1 - pi**2, 0, 1) + 1e-6).log().sum(1)
        return mu, pi, logp_pi

    def copy_param(self):
        self.v_targ.load_state_dict(self.v.state_dict())

    def get_action(self, obs):
        mu, pi, logp_pi = self.policy(obs)
        mu, pi, logp_pi = self.squashing(mu, pi, logp_pi)
        mu = self.act_limit * mu
        pi = self.act_limit * pi
        return mu, pi, logp_pi

    def update_target(self, rho):
        # compute rho * targ_p + (1 - rho) * main_p
        for v_p, v_targ_p in zip(self.v.parameters(), self.v_targ.parameters()):
            v_targ_p.data = rho * v_targ_p.data + (1-rho) * v_p.data

    def compute_v_target(self, obs, alpha):
        _, pi, logp = self.get_action(obs)
        q1, q2 = self.q1(torch.cat([obs, pi], 1)), self.q2(torch.cat([obs, pi], 1))
        q = torch.min(q1, q2).squeeze()
        return (q - alpha * logp.squeeze()).detach()

    def compute_q_target(self, obs, gamma, rewards, done):
        # compute r + gamma * (1 - d) * V(s')
        return (rewards + gamma * (1-done) * self.v_targ(obs).squeeze()).detach()

    def q_function(self, obs, pi):
        q1, q2 = self.q1(torch.cat([obs, pi], 1)), self.q2(torch.cat([obs, pi], 1))
        return q1.squeeze(), q2.squeeze()

    def q_function_w_entropy(self, obs, alpha):
        _, pi, logp_pi = self.get_action(obs)
        q1 = self.q1(torch.cat([obs, pi], 1)).squeeze()
        H = -logp_pi * alpha
        return q1 + H.squeeze()