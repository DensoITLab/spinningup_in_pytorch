import numpy as np
import gym
import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import scipy.signal
import functools
from tensorboardX import SummaryWriter


# Neural network to train value function
class ValueNetwork(nn.Module):

    def __init__(self):
        super(ValueNetwork, self).__init__()
        self.fc1 = nn.Linear(4, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# Neural network to train policy function
class PolicyNetwork(nn.Module):

    def __init__(self):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(4, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 2)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.softmax(self.fc3(x))
        return x


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


if __name__ == '__main__':

    env = gym.make('CartPole-v0')
    env.reset()

    # hyper parameter
    epochs = 3000
    steps_per_epoch = 4000

    policy_network_learning_rate = 3e-4
    value_network_learning_rate = 1e-3

    train_value_network_iteration = 80

    gamma = 0.99
    lambd = 0.95

    # initialize network
    policy_network = PolicyNetwork()
    value_network = ValueNetwork()

    optimizer_policy = optim.Adam(policy_network.parameters(), lr=policy_network_learning_rate)
    optimizer_value = optim.Adam(value_network.parameters(), lr=value_network_learning_rate)

    writer = SummaryWriter()

    all_episodes = 1

    d = datetime.datetime.now()
    time_to_start_training = "{0:%Y%m%d_%H%M%S}".format(d)

    for i in range(epochs):

        print("Epoch %d" % i)

        # initialize variables per epoch
        observations = []
        actions = []
        returns = []
        advantages = []

        step_in_epoch = 0

        while True:
            observations_per_episode = []
            actions_per_episode = []
            values_per_episode = []
            rewards_per_episode = []

            reward = 0.0

            # observation = (position, velocity, rotation, rotaion speed)
            observation = env.reset()

            episode_length = 0

            # start episode
            while True:
                step_in_epoch += 1
                episode_length += 1

                # inference action probability and value
                x = torch.from_numpy(observation.astype(np.float32))
                action_probability = policy_network(x)
                value = value_network(x)

                # sampling action according to action probability
                action = torch.multinomial(action_probability, 1)

                # rendering
                if i > 30:
                    env.render()

                # save observation, action, value, reward
                observations_per_episode.append(observation)
                actions_per_episode.append(action)
                values_per_episode.append(value.item())
                rewards_per_episode.append(reward)

                # action
                (observation, reward, done, info) = env.step(action.item())

                if step_in_epoch > steps_per_epoch:
                    done = True

                if done:
                    break

            # append last value to rewards and values
            # in order to calculate differences of rewards and values.
            rewards_per_episode.append(rewards_per_episode[len(rewards_per_episode) - 1])
            values_per_episode.append(rewards_per_episode[len(rewards_per_episode) - 1])

            # tensorboard
            writer.add_scalar('length', episode_length, all_episodes)
            all_episodes += 1

            # GAE-Lambda advantage calculation
            # High Dimensional Continuous Control Using Generalized Advantage Estimation, Schulman et al. 2016(b)
            temp_rewards = np.array(rewards_per_episode)
            temp_values = np.array(values_per_episode)
            deltas = temp_rewards[:-1] + gamma * temp_values[1:] - temp_values[:-1]

            # save
            observations.append(observations_per_episode)
            actions.append(actions_per_episode)
            advantages.append(discount_cumsum(deltas, gamma * lambd).tolist())
            returns.append(discount_cumsum(rewards_per_episode, gamma)[:-1].tolist())

            if step_in_epoch > steps_per_epoch:
                break

        def compact_array_to_torch_float(l):
            temp = functools.reduce(lambda x, y: x + y, l)
            return torch.from_numpy(np.array(temp)).float()

        def compact_array_to_torch_long(l):
            temp = functools.reduce(lambda x, y: x + y, l)
            return torch.from_numpy(np.array(temp)).long()

        def update_policy_network(observations, actions, advantages):
            # normalize advantages
            mu = advantages.mean()
            dev = advantages.std()
            advantages = (advantages - mu) / dev

            # learning policy network
            optimizer_policy.zero_grad()
            action_probability = policy_network(observations)

            onehot = torch.eye(2)[actions]
            likelihood_for_history = (onehot * action_probability.log()).sum(1)

            loss = - torch.mean(likelihood_for_history * advantages)
            loss.backward()
            optimizer_policy.step()

        def update_value_network(observations, returns):
            # learning value network
            for _ in range(train_value_network_iteration):
                v = value_network(observations)
                optimizer_value.zero_grad()
                loss = F.mse_loss(returns, v)
                loss.backward()
                optimizer_value.step()

        observations = compact_array_to_torch_float(observations)
        actions = compact_array_to_torch_long(actions)
        advantages = compact_array_to_torch_float(advantages)
        returns = compact_array_to_torch_float(returns)

        update_policy_network(observations, actions, advantages)
        update_value_network(observations, returns)

        torch.save(value_network.state_dict(), './model/value_network_model_%s_%08d.model' % (time_to_start_training, i))
        torch.save(policy_network.state_dict(), './model/policy_network_model_%s_%08d.model' % (time_to_start_training, i))

    env.env.close()
