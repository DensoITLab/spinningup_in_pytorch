import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import numpy as np
import time
import gym

import core
from spinup.utils.logx import EpochLogger

class PPOBuffer:
    def __init__(self):
        self.init_param()

    def init_param(self):
        self.obs_buf = []
        self.act_buf = []
        self.adv_buf = []
        self.rew_buf = []
        self.ret_buf = []
        self.val_buf = []
        self.logp_buf = []

    def store(self, ep_obs, ep_act, ep_rew, ep_val, ep_logp, adv, ret):
        self.obs_buf += ep_obs
        self.act_buf += ep_act
        self.rew_buf += ep_rew
        self.val_buf += ep_val
        self.logp_buf += ep_logp
        self.adv_buf += list(adv)
        self.ret_buf += list(ret)

    def get(self):
        adv_tns = torch.FloatTensor(self.adv_buf)
        adv_tns = (adv_tns - adv_tns.mean()) / adv_tns.std()
        return_tns = [
            torch.FloatTensor(self.obs_buf),
            torch.FloatTensor(self.act_buf),
            torch.FloatTensor(self.logp_buf),
            torch.FloatTensor(self.ret_buf),
            adv_tns
        ]
        self.init_param()
        return return_tns

def ppo(env_name, actor_critic_func, hidden_sizes, 
        steps_per_epoch=4000, epochs=50, gamma=0.99, clip_ratio=0.2, pi_lr=3e-4,
        vf_lr=1e-3, train_pi_iters=80, train_v_iters=80, lam=0.97, max_ep_len=1000,
        target_kl=0.01, logger_kwargs=dict()):

    logger = EpochLogger(**logger_kwargs)
    logger.save_config(locals())

    env = gym.make(env_name)

    buf = PPOBuffer()

    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n

    actor_critic = actor_critic_func(obs_dim, act_dim, hidden_sizes)

    value_optim  = optim.Adam(actor_critic.value_function.parameters(), lr=vf_lr)
    policy_optim = optim.Adam(actor_critic.policy.parameters(), lr=pi_lr)

    def finish_path(ep_rew, ep_val, last_value):
        rews = np.array(ep_rew + [last_value])
        vals = np.array(ep_val + [last_value])
        deltas = rews[:-1] + gamma * vals[1:] - vals[:-1]

        adv = core.discount_cumsum(deltas, gamma * lam)
        ret = core.discount_cumsum(rews, gamma)[:-1]

        return adv, ret

    def update():
        obs_tns, act_tns, old_logp, ret_tns, adv_tns = buf.get()
        
        for i in range(train_pi_iters):
            logp = actor_critic.likelihood(obs_tns, act_tns)
            ratio = (logp - old_logp).exp()
            min_adv = torch.where(adv_tns > 0, (1*clip_ratio)*adv_tns, (1-clip_ratio*adv_tns))
            pi_loss = -(torch.min(ratio * adv_tns, min_adv)).mean()
            policy_optim.zero_grad()
            pi_loss.backward()
            policy_optim.step()

            kl = (old_logp - logp).mean()
            if kl > 1.5 * target_kl:
                logger.log('Early stopping at step %d due to reaching max kl.'%i)
                break
        logger.store(StopIter=i)
        for _ in range(train_v_iters):
            v = actor_critic.value_function(obs_tns)
            v_loss = (ret_tns - v).pow(2).mean()
            value_optim.zero_grad()
            v_loss.backward()
            value_optim.step()
        logger.store(LossPi=pi_loss, LossV=v_loss, KL=kl)

    start_time = time.time()
    obs, rew, done, ep_ret, ep_len = env.reset(), 0, False, 0, 0
    ep_obs, ep_act, ep_rew, ep_val, ep_logp = [], [], [], [], []

    for epoch in range(epochs):
        for t in range(steps_per_epoch):
            obs_tns = torch.from_numpy(obs).reshape(1,-1).float()
            act, logp = actor_critic.sampling_action(obs_tns)
            val = actor_critic.value_function(obs_tns).item()

            ep_obs.append(obs); ep_act.append(act)
            ep_rew.append(rew); ep_val.append(val)
            ep_logp.append(logp)

            logger.store(VVals=val)

            obs, rew, done, _ = env.step(act)
            ep_ret += rew
            ep_len += 1

            terminal = done or (ep_len == max_ep_len)
            if terminal or (t==steps_per_epoch-1):
                if not(terminal):
                    print('Warning: trajectory cut off by epoch at %d steps.'%ep_len)
                if done:
                    last_val = rew
                else:
                    obs_tns = torch.from_numpy(obs).reshape(1,-1).float()
                    last_val = actor_critic.value_function(obs_tns)
                adv, ret = finish_path(ep_rew, ep_val, last_val)
                if terminal:
                    logger.store(EpRet=ep_ret, EpLen=ep_len)
                buf.store(ep_obs, ep_act, ep_rew, ep_val, ep_logp, adv, ret)
                obs, rew, done, ep_ret, ep_len = env.reset(), 0, False, 0, 0
                ep_obs, ep_act, ep_rew, ep_val, ep_logp = [], [], [], [], []

        update()

        logger.log_tabular('Epoch', epoch)
        logger.log_tabular('EpRet', with_min_and_max=True)
        logger.log_tabular('EpLen', average_only=True)
        logger.log_tabular('VVals', with_min_and_max=True)
        logger.log_tabular('LossPi', average_only=True)
        logger.log_tabular('LossV', average_only=True)
        logger.log_tabular('KL', average_only=True)
        logger.log_tabular('StopIter', average_only=True)
        logger.log_tabular('Time', time.time()-start_time)
        logger.dump_tabular()

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='CartPole-v0')
    parser.add_argument('--hid', type=int, default=64)
    parser.add_argument('--l', type=int, default=2)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--steps', type=int, default=4000)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--exp_name', type=str, default='ppo')
    args = parser.parse_args()

    from spinup.utils.run_utils import setup_logger_kwargs
    logger_kwargs = setup_logger_kwargs(args.exp_name, args.seed)

    ppo(args.env, core.actor_critic,
        [args.hid]*args.l, gamma=args.gamma, 
        steps_per_epoch=args.steps, epochs=args.epochs,
        logger_kwargs=logger_kwargs)