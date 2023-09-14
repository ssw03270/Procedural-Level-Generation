import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torch.distributions import Categorical
import copy
import math
import yaml

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_config():
    path = 'config.yaml'
    with open(path, 'r') as stream:
        return yaml.load(stream, Loader=yaml.FullLoader)

def get_mask(data):
    config = get_config()

    pattern = torch.tensor([0, 0, 0, 0, 0, 0, 0, 0]).to(device)

    mask = ~(data == pattern).all(dim=2)
    count = torch.sum(mask, -1, keepdim=True)

    mask = mask.unsqueeze(-1)
    mask = mask.repeat(1, 1, config['feature_size'])

    return mask, count

class Attention(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = get_config()

        self.wq = nn.Linear(self.config['feature_size'], self.config['feature_size'])
        self.wq.requires_grad = True

        self.wk = nn.Linear(self.config['feature_size'], self.config['feature_size'])
        self.wk.requires_grad = True

    def forward(self, query, key):
        # query 는 graph (batch, graph_feature)
        # key 는 rule (batch, rule_count, rule_feature)
        query = self.wq(query)
        key = self.wk(key)

        score = torch.matmul(query / math.sqrt(self.config['feature_size']), key.transpose(1, 2))
        attn = F.softmax(score, dim=2)

        return attn

class Actor(nn.Module):
    def __init__(self):
        super(Actor, self).__init__()

        self.config = get_config()
        self.attn = Attention()

        self.aggregate_layer = nn.Linear(self.config['edge_size'], self.config['feature_size'])
        self.relu = F.relu

        self.terminate_layer = nn.Linear(self.config['rule_max'], 2)
        self.prob_layer = nn.Linear(self.config['rule_max'] * 2, self.config['rule_max'] * 2)

    def forward(self, state):
        state = state.view(-1, self.config['level_max'] + self.config['rule_max'], 8)

        h = self.aggregate_layer(state)
        h = self.relu(h)

        level_mask, level_count = get_mask(state[:, :self.config['level_max']])
        level = h[:, :self.config['level_max']] * level_mask
        level = torch.sum(level, dim=1) / level_count
        level = level.view(-1, 1, self.config['feature_size'])

        rule_mask, rule_count = get_mask(state[:, self.config['level_max']:])
        rules = h[:, self.config['level_max']:] * rule_mask
        generate_score = self.attn(level, rules)
        terminate_score = F.softmax(self.terminate_layer(generate_score), dim=2)

        g = torch.cat((generate_score.view(-1, self.config['rule_max'], 1), generate_score.view(-1, self.config['rule_max'], 1)), dim=2)
        t = terminate_score.view(-1, 1, 2)

        act = g * t
        act = act.view(-1, self.config['rule_max'] * 2)

        return act

    def pi(self, state):
        act = self.forward(state)
        prob = F.softmax(self.prob_layer(act), dim=1)
        return prob

    # def get_dist(self,state):
    #     alpha,beta = self.forward(state)
    #     dist = Beta(alpha, beta)
    #     return dist


class Critic(nn.Module):
    def __init__(self):
        super(Critic, self).__init__()

        self.config = get_config()
        self.attn = Attention()

        self.aggregate_layer = nn.Linear(self.config['edge_size'], self.config['feature_size'])
        self.relu = F.relu

        self.terminate_layer = nn.Linear(self.config['rule_max'], 2)

        self.critic_layer = nn.Linear(self.config['rule_max'] * 2, 1)

    def forward(self, state):
        state = state.view(-1, self.config['level_max'] + self.config['rule_max'], 8)

        h = self.aggregate_layer(state)
        h = self.relu(h)

        level_mask, level_count = get_mask(state[:, :self.config['level_max']])
        level = h[:, :self.config['level_max']] * level_mask
        level = torch.sum(level, dim=1) / level_count
        level = level.view(-1, 1, self.config['feature_size'])

        rule_mask, rule_count = get_mask(state[:, self.config['level_max']:])
        rules = h[:, self.config['level_max']:]
        rules = rules.view(-1, self.config['rule_max'], self.config['feature_size'])

        generate_score = self.attn(level, rules)
        terminate_score = F.softmax(self.terminate_layer(generate_score), dim=2)

        g = torch.cat((generate_score.view(-1, self.config['rule_max'], 1), generate_score.view(-1, self.config['rule_max'], 1)), dim=2)
        t = terminate_score.view(-1, 1, 2)

        act = g * t
        act = act.view(-1, self.config['rule_max'] * 2)

        value = self.critic_layer(act)
        return value


class PPO_discrete(object):
    def __init__(
            self,
            gamma=0.99,
            lambd=0.95,
            lr=1e-4,
            clip_rate=0.2,
            K_epochs=10,
            batch_size=64,
            l2_reg=1e-3,
            entropy_coef=1e-3,
            adv_normalization=False,
            entropy_coef_decay=0.99,
    ):

        self.actor = Actor().to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr)

        self.critic = Critic().to(device)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr)

        self.data = []
        self.gamma = gamma
        self.lambd = lambd
        self.clip_rate = clip_rate
        self.K_epochs = K_epochs
        self.optim_batch_size = batch_size
        self.l2_reg = l2_reg
        self.entropy_coef = entropy_coef
        self.adv_normalization = adv_normalization
        self.entropy_coef_decay = entropy_coef_decay

    def select_action(self, state):
        '''Stochastic Policy'''
        with torch.no_grad():
            pi = self.actor.pi(state)
            m = Categorical(pi)
            a = m.sample().item()
            pi_a = pi[:, a].item()
        return a, pi_a

    def evaluate(self, state):
        '''Deterministic Policy'''
        with torch.no_grad():
            pi = self.actor.pi(state)
            a = torch.argmax(pi).item()
        return a, 1.0

    def train(self):
        s, a, r, s_prime, old_prob_a, done_mask, dw_mask = self.make_batch()
        self.entropy_coef *= self.entropy_coef_decay  # exploring decay

        ''' Use TD+GAE+LongTrajectory to compute Advantage and TD target'''
        with torch.no_grad():
            vs = self.critic(s)
            vs_ = self.critic(s_prime)

            '''dw(dead and win) for TD_target and Adv'''
            deltas = r + self.gamma * vs_ * (1 - dw_mask) - vs
            deltas = deltas.cpu().flatten().numpy()
            adv = [0]

            '''done for GAE'''
            for dlt, mask in zip(deltas[::-1], done_mask.cpu().flatten().numpy()[::-1]):
                advantage = dlt + self.gamma * self.lambd * adv[-1] * (1 - mask)
                adv.append(advantage)
            adv.reverse()
            adv = copy.deepcopy(adv[0:-1])
            adv = torch.tensor(adv).unsqueeze(1).float().to(device)
            td_target = adv + vs
            if self.adv_normalization:
                adv = (adv - adv.mean()) / ((adv.std() + 1e-4))  # useful in some envs

        """PPO update"""
        # Slice long trajectopy into short trajectory and perform mini-batch PPO update
        optim_iter_num = int(math.ceil(s.shape[0] / self.optim_batch_size))

        for _ in range(self.K_epochs):
            # Shuffle the trajectory, Good for training
            perm = np.arange(s.shape[0])
            np.random.shuffle(perm)
            perm = torch.LongTensor(perm).to(device)
            s, a, td_target, adv, old_prob_a = \
                s[perm].clone(), a[perm].clone(), td_target[perm].clone(), adv[perm].clone(), old_prob_a[perm].clone()

            '''mini-batch PPO update'''
            for i in range(optim_iter_num):
                index = slice(i * self.optim_batch_size, min((i + 1) * self.optim_batch_size, s.shape[0]))

                '''actor update'''
                prob = self.actor.pi(s[index])

                entropy = Categorical(prob).entropy().sum(0, keepdim=True)
                prob_a = prob.gather(1, a[index])
                ratio = torch.exp(torch.log(prob_a) - torch.log(old_prob_a[index]))  # a/b == exp(log(a)-log(b))

                surr1 = ratio * adv[index]
                surr2 = torch.clamp(ratio, 1 - self.clip_rate, 1 + self.clip_rate) * adv[index]
                a_loss = -torch.min(surr1, surr2) - self.entropy_coef * entropy

                self.actor_optimizer.zero_grad()
                a_loss.mean().backward()
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 40)
                self.actor_optimizer.step()

                '''critic update'''
                c_loss = (self.critic(s[index]) - td_target[index]).pow(2).mean()
                for name, param in self.critic.named_parameters():
                    if 'weight' in name:
                        c_loss += param.pow(2).sum() * self.l2_reg

                self.critic_optimizer.zero_grad()
                c_loss.backward()
                self.critic_optimizer.step()
        return a_loss.mean(), c_loss, entropy

    def make_batch(self):
        l = len(self.data)
        s_lst, a_lst, r_lst, s_prime_lst, prob_a_lst, done_lst, dw_lst = \
            np.zeros((l, 1, 1600)), np.zeros((l, 1)), np.zeros((l, 1)), np.zeros((l, 1, 1600)), np.zeros(
                (l, 1)), np.zeros((l, 1)), np.zeros((l, 1))

        for i, transition in enumerate(self.data):
            s_lst[i], a_lst[i], r_lst[i], s_prime_lst[i], prob_a_lst[i], done_lst[i], dw_lst[i] = transition

        self.data = []  # Clean history trajectory

        '''list to tensor'''
        with torch.no_grad():
            s, a, r, s_prime, prob_a, done_mask, dw_mask = \
                torch.tensor(s_lst, dtype=torch.float).to(device), \
                    torch.tensor(a_lst, dtype=torch.int64).to(device), \
                    torch.tensor(r_lst, dtype=torch.float).to(device), \
                    torch.tensor(s_prime_lst, dtype=torch.float).to(device), \
                    torch.tensor(prob_a_lst, dtype=torch.float).to(device), \
                    torch.tensor(done_lst, dtype=torch.float).to(device), \
                    torch.tensor(dw_lst, dtype=torch.float).to(device),

        return s, a, r, s_prime, prob_a, done_mask, dw_mask

    def put_data(self, transition):
        self.data.append(transition)

    def save(self, episode):
        torch.save(self.critic.state_dict(), "./model/ppo_critic{}.pth".format(episode))
        torch.save(self.actor.state_dict(), "./model/ppo_actor{}.pth".format(episode))

    def load(self, episode):
        self.critic.load_state_dict(torch.load("./model/ppo_critic{}.pth".format(episode)))
        self.actor.load_state_dict(torch.load("./model/ppo_actor{}.pth".format(episode)))