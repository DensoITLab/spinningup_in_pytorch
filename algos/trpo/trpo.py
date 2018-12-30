import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import numpy as np
import time
import gym

from core import actor_critic as ac
import core
from spinup.utils.logx import EpochLogger

EPS = 1e-8

class GAEBuffer:
    def __init__(self):
        self.buf_init()

    def buf_init(self):
        self.obs_buf = []
        self.act_buf = []
        self.adv_buf = []
        self.rew_buf = []
        self.ret_buf = []
        self.val_buf = []
        self.logp_buf = []

    def store(self, ep_obs, ep_act, ep_rew, ep_val, adv, ret):
        self.obs_buf += ep_obs
        self.act_buf += ep_act
        self.rew_buf += ep_rew
        self.val_buf += ep_val
        self.adv_buf += list(adv)
        self.ret_buf += list(ret)

    def get(self):
        adv = torch.FloatTensor(self.adv_buf)
        adv_mean, adv_std = adv.mean(), adv.std()
        adv = (adv - adv_mean)/adv_std
        return_list = [
            torch.FloatTensor(self.obs_buf),
            torch.FloatTensor(self.act_buf),
            torch.FloatTensor(self.ret_buf),
            adv
        ]
        self.buf_init()
        return return_list

def finish_path(ep_rew, ep_val, last_value, gamma, lam):
    rews = np.array(ep_rew + [last_value])
    vals = np.array(ep_val + [last_value])
    deltas = rews[:-1] + gamma * vals[1:] - vals[:-1]

    adv = core.discount_cumsum(deltas, gamma * lam)
    ret = core.discount_cumsum(rews, gamma)[:-1]

    return adv, ret

def trpo(env_name, actor_critic_func, hidden_sizes,
        steps_per_epoch=4000, epochs=50, gamma=0.99,
        delta=0.01, vf_lr=1e-3, train_v_iters=80,
        damping_coeff=0.1, cg_iters=10, backtrack_iters=10,
        backtrack_coeff=0.8, lam=0.97, max_ep_len=1000,
        logger_kwargs=dict()):

    logger = EpochLogger(**logger_kwargs)

    buf = GAEBuffer()

    env = gym.make(env_name)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n

    actor_critic = actor_critic_func(act_dim, obs_dim, hidden_sizes)

    value_optim = optim.Adam(actor_critic.value_function.parameters(), lr=vf_lr)

    start_time = time.time()
    obs, rew, done, ep_ret, ep_len = env.reset(), 0, False, 0, 0
    ep_obs, ep_act, ep_rew, ep_val = [], [], [], []

    def cg(Hx, b):
        x = torch.zeros_like(b)
        r = b.clone()
        p = r.clone()
        r_dot_old = torch.dot(r, r)
        for _ in range(cg_iters):
            z = Hx(p)
            z = z + damping_coeff * p
            alpha = r_dot_old / (torch.dot(p, z) + EPS)
            x += alpha * p
            r -= alpha * z
            r_dot_new = torch.dot(r, r)
            p = r + (r_dot_new / r_dot_old) * p
            r_dot_old = r_dot_new
        return x

    def update():
        obs_tns, act_tns, ret_tns, adv_tns = buf.get()
        # log likelihood for policy and old policy
        logp_all, logp = actor_critic.likelihood(obs_tns, act_tns)
        old_logp_all, old_logp = actor_critic.old_likelihood(obs_tns, act_tns)
        # kl div
        d_kl = actor_critic.policy.kl(logp_all, old_logp_all.detach())
        # gradient
        ratio = (logp - old_logp.detach()).exp()
        pi_loss = -(ratio * adv_tns).mean()

        pi_gradients = [torch.autograd.grad(pi_loss, p, retain_graph=True)[0] \
                        for p in actor_critic.policy.parameters()]
        Hx = lambda x: core.hessian_vector_product(x, d_kl, actor_critic.policy)
        x = cg(Hx, core.params_flat(pi_gradients))
        alpha = (2*delta/(torch.dot(x, Hx(x) + damping_coeff * x) + EPS)).sqrt()

        old_params = core.params_flat(actor_critic.old_policy.parameters())

        for j in range(backtrack_iters):
            actor_critic.policy_update(old_params-alpha*x*backtrack_coeff**j)
            logp_all, logp = actor_critic.likelihood(obs_tns, act_tns)
            d_kl = actor_critic.policy.kl(logp_all, old_logp_all.detach())
            ratio = (logp - old_logp.detach()).exp()
            pi_loss_new = -(ratio * adv_tns).mean()

            if d_kl <= delta and pi_loss_new <= pi_loss:
                logger.log('Accepting new params at step %d of line search.'%j)
                logger.store(BacktrackIters=j)
                break

            if j == backtrack_iters-1:
                actor_critic.policy_update(old_params)
                logger.log('Line search failed! Keeping old params.')
                logger.store(BacktrackIters=j)

        for _ in range(train_v_iters):
            v = actor_critic.value_function(obs_tns).squeeze()
            v_loss = (ret_tns - v).pow(2).mean()
            value_optim.zero_grad()
            v_loss.backward()
            value_optim.step()

        actor_critic.copy_policy_param()

    for epoch in range(epochs):
        for t in range(steps_per_epoch):
            obs_tens = torch.from_numpy(obs).float().reshape(1,-1)

            action = actor_critic.sampling_action(obs_tens).item()
            value  = actor_critic.value_function(obs_tens).squeeze().item()

            ep_obs.append(obs); ep_act.append(action)
            ep_rew.append(rew), ep_val.append(value)
            logger.store(VVals=value)

            obs, rew, done, _ = env.step(action)
            ep_ret += rew
            ep_len += 1

            terminal = done or (ep_len == max_ep_len)
            if terminal or (t==steps_per_epoch-1):
                if not(terminal):
                    print('Warning: trajectory cut off by epoch at %d steps.'%ep_len)
                if done:
                    last_value = rew
                else:
                    obs_tens = torch.from_numpy(obs).float().reshape(1,-1)
                    last_value = actor_critic.value_function(obs_tens).squeeze().item()
                adv, ret = finish_path(ep_rew, ep_val, last_value, gamma, lam)
                buf.store(ep_obs, ep_act, ep_rew, ep_val, adv, ret)
                if terminal:
                    logger.store(EpRet=ep_ret, EpLen=ep_len)
                obs, rew, done, ep_ret, ep_len = env.reset(), 0, False, 0, 0
                ep_obs, ep_act, ep_rew, ep_val = [], [], [], []
        update()

        # Log info about epoch
        logger.log_tabular('Epoch', epoch)
        logger.log_tabular('EpRet', with_min_and_max=True)
        logger.log_tabular('EpLen', average_only=True)
        logger.log_tabular('VVals', with_min_and_max=True)
        logger.log_tabular('Time', time.time()-start_time)
        logger.dump_tabular()

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='CartPole-v0')
    parser.add_argument('--hid', type=int, default=64)
    parser.add_argument('--l', type=int, default=2)
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--steps', type=int, default=4000)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--exp_name', type=str, default='trpo')
    args = parser.parse_args()

    from spinup.utils.run_utils import setup_logger_kwargs
    logger_kwargs = setup_logger_kwargs(args.exp_name, args.seed)

    trpo(args.env, ac,
         [args.hid]*args.l, gamma=args.gamma, 
        steps_per_epoch=args.steps, epochs=args.epochs,
         logger_kwargs=logger_kwargs)