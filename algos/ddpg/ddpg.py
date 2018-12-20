import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import gym, time
import numpy as np

from spinup.utils.logx import EpochLogger
from core import actor_critic as ac

class ReplayBuffer:
    def __init__(self, size):
        self.size, self.max_size = 0, size
        self.obs1_buf = []
        self.obs2_buf = []
        self.acts_buf = []
        self.rews_buf = []
        self.done_buf = []

    def store(self, obs, act, rew, next_obs, done):
        self.obs1_buf.append(obs)
        self.obs2_buf.append(next_obs)
        self.acts_buf.append(act)
        self.rews_buf.append(rew)
        self.done_buf.append(int(done))
        while len(self.obs1_buf) > self.max_size:
            self.obs1_buf.pop(0)
            self.obs2_buf.pop(0)
            self.acts_buf.pop(0)
            self.rews_buf.pop(0)
            self.done_buf.pop(0)

        self.size = len(self.obs1_buf)

    def sample_batch(self, batch_size=32):
        idxs = np.random.randint(low=0, high=self.size, size=(batch_size,))
        obs1 = torch.FloatTensor([self.obs1_buf[i] for i in idxs])
        obs2 = torch.FloatTensor([self.obs2_buf[i] for i in idxs])
        acts = torch.FloatTensor([self.acts_buf[i] for i in idxs])
        rews = torch.FloatTensor([self.rews_buf[i] for i in idxs])
        done = torch.FloatTensor([self.done_buf[i] for i in idxs])
        return [obs1, obs2, acts, rews, done]

def ddpg(env_name, actor_critic_function, hidden_size,
        steps_per_epoch=5000, epochs=100, replay_size=int(1e6), gamma=0.99, 
        polyak=0.995, pi_lr=1e-3, q_lr=1e-3, batch_size=100, start_steps=10000, 
        act_noise=0.1, max_ep_len=1000, logger_kwargs=dict()):

    logger = EpochLogger(**logger_kwargs)
    logger.save_config(locals())

    replay_buffer = ReplayBuffer(replay_size)

    env, test_env = gym.make(env_name), gym.make(env_name)

    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    act_limit = int(env.action_space.high[0])

    actor_critic = actor_critic_function(act_dim, obs_dim, hidden_size, act_limit)

    q_optimizer = optim.Adam(actor_critic.q.parameters(), q_lr)
    policy_optimizer = optim.Adam(actor_critic.policy.parameters(), pi_lr)

    start_time = time.time()

    obs, ret, done, ep_ret, ep_len = env.reset(), 0, False, 0, 0
    total_steps = steps_per_epoch * epochs

    for t in range(total_steps):
        if t > 50000:
            env.render()
        if t > start_steps:
            obs_tens = torch.from_numpy(obs).float().reshape(1,-1)
            act = actor_critic.get_action(obs_tens, act_noise).detach().numpy().reshape(-1)
        else:
            act = env.action_space.sample()

        obs2, ret, done, _ = env.step(act)

        ep_ret += ret
        ep_len += 1

        done = False if ep_len==max_ep_len else done

        replay_buffer.store(obs, act, ret, obs2, done)

        obs = obs2

        if done or (ep_len == max_ep_len):
            for _ in range(ep_len):
                obs1_tens, obs2_tens, acts_tens, rews_tens, done_tens = replay_buffer.sample_batch(batch_size)
                # compute Q(s, a)
                q = actor_critic.q_function(obs1_tens, action=acts_tens)
                # compute r + gamma * (1 - d) * Q(s', mu_targ(s'))
                q_targ = actor_critic.compute_target(obs2_tens, gamma, rews_tens, done_tens)
                # compute (Q(s, a) - y(r, s', d))^2
                q_loss = (q-q_targ).pow(2).mean()

                q_optimizer.zero_grad()
                q_loss.backward()
                q_optimizer.step()

                logger.store(LossQ=q_loss.item(), QVals=q.detach().numpy())

                # compute Q(s, mu(s))
                policy_loss = -actor_critic.q_function(obs1_tens, detach=False).mean()
                policy_optimizer.zero_grad()
                policy_loss.backward()
                policy_optimizer.step()

                logger.store(LossPi=policy_loss.item())

                # compute rho * targ_p + (1 - rho) * main_p
                actor_critic.update_target(polyak)

            logger.store(EpRet=ep_ret, EpLen=ep_len)
            obs, ret, done, ep_ret, ep_len = env.reset(), 0, False, 0, 0

        if t > 0 and t % steps_per_epoch == 0:
            epoch = t // steps_per_epoch

            # test_agent()
            logger.log_tabular('Epoch', epoch)
            logger.log_tabular('EpRet', with_min_and_max=True)
            # logger.log_tabular('TestEpRet', with_min_and_max=True)
            logger.log_tabular('EpLen', average_only=True)
            # logger.log_tabular('TestEpLen', average_only=True)
            logger.log_tabular('TotalEnvInteracts', t)
            logger.log_tabular('QVals', with_min_and_max=True)
            logger.log_tabular('LossPi', average_only=True)
            logger.log_tabular('LossQ', average_only=True)
            logger.log_tabular('Time', time.time()-start_time)
            logger.dump_tabular()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='Pendulum-v0')
    parser.add_argument('--hid', type=int, default=300)
    parser.add_argument('--l', type=int, default=1)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--exp_name', type=str, default='ddpg')
    args = parser.parse_args()

    from spinup.utils.run_utils import setup_logger_kwargs
    logger_kwargs = setup_logger_kwargs(args.exp_name, args.seed)

    ddpg(args.env, actor_critic_function=ac,
        hidden_size=[args.hid]*args.l, gamma=args.gamma, epochs=args.epochs,
        logger_kwargs=logger_kwargs)