import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal

class PI_GPS(object):

    def __init__(self, envs, policy, T, lam = 0.1):
        self.lam = 0.1
        self.envs = envs
        self.policy = policy
        self.T = T
        self.N = envs.num_envs
        self.action_dim = self.envs.action_space.shape[0]
        self.state_dim = self.envs.observation_space.shape[0]
        self.feed_forward = torch.zeros(T, self.action_dim)
        self.feed_forward_var = torch.ones(T, self.action_dim)

        self.optim = optim.Adam(self.policy.parameters(), lr=3e-3)

    def update(self):
        with torch.no_grad():
            s = self.envs.reset()
            states, actions = [], []
            sk, eps, ks = [], [], []
            rew = 0.
            for t in range(self.T):
                s = torch.FloatTensor(s)
                states.append(s)
                var = self.feed_forward_var[t].repeat(self.N, 1)
                eps.append(var * torch.randn_like(var))
                k = self.feed_forward[t] + eps[-1]
                ks.append(k)
                mu, log_var = self.policy(s)
                v = mu + k
                actions.append(v)
                s, r, d, _ = self.envs.step(v.numpy())
                rew += r
                sk.append(torch.FloatTensor(r).squeeze())

            states = torch.cat(states, axis=0)
            # actions = torch.cat(actions, axis=0)
            sk = torch.stack(sk)
            sk = torch.cumsum(sk.flip(0), 0).flip(0)

            sk = sk - torch.max(sk, dim=1, keepdim=True)[0]
            w = torch.exp(sk.div(self.lam)) + 1e-3
            w.div_(torch.sum(w, dim=1, keepdim=True))

            for t in range(self.T):
                self.feed_forward[t] = torch.mv(ks[t].T, w[t])
                dk = self.feed_forward[t] - ks[t]
                # self.feed_forward_var[t] = torch.mv(torch.pow(dk.T,2), w[t])
        mu, log_std = self.policy(states)

        k = self.feed_forward.repeat(1, self.N).view(self.N*self.T, -1)

        pi = Normal(mu, log_std.exp())
        loss = -torch.mean(pi.log_prob((mu+k).detach()))

        self.optim.zero_grad()
        loss.backward()
        self.optim.step()
        print(loss.item(), np.mean(rew))
