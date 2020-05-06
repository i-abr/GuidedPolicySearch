import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal

class GuidedPolicySearch(object):

    def __init__(self, model, policy, T=10, lr=0.1, policy_lr=3e-3):

        self.T = T
        self.model = model
        self.policy = policy
        self.state_dim = model.num_states
        self.action_dim = model.num_actions

        self.device = 'cpu'
        if torch.cuda.is_available():
            self.device = 'cuda:0'


        # iLQR stuff
        self.lr     = lr
        self.K      = torch.randn(T, self.action_dim, self.state_dim).to(self.device)
        self.k      = torch.randn(T, self.action_dim).to(self.device)
        self.xbar   = torch.randn(T, self.state_dim).to(self.device)

        self.K.requires_grad    = True
        self.xbar.requires_grad = True
        self.k.requires_grad    = True
        self.ilqr_optim = optim.Adam([self.K, self.k, self.xbar], lr=lr)

        # policy cosntraint stuff
        self.lam = torch.zeros(T, self.action_dim).to(self.device)
        self.rho = 1e-2
        self.policy_optim = optim.Adam(self.policy.parameters(), lr=policy_lr)

    def reset(self):
        with torch.no_grad():
            self.K.zero_()
            self.k.zero_()
            self.xbar.zero_()

    def update_constraint(self, state):
        with torch.no_grad():
            s = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            cost = 0.
            for i, (K, k, xbar, lam) in enumerate(zip(self.K, self.k, self.xbar, self.lam)):
                mu, _ = self.policy(s)
                u = torch.mv(K, xbar - s.squeeze()) + k
                s, r = self.model.step(s, u.unsqueeze(0))
                self.lam[i] += self.rho * (torch.tanh(mu.squeeze()) - u)

    def update(self, state):
        self.update_constraint(state)
        s = torch.FloatTensor(state).unsqueeze(0).to(self.device)

        cost = 0.
        for K, k, xbar in zip(self.K, self.k, self.xbar):
            mu, _ = self.policy(s)
            u = torch.mv(K, xbar - s.squeeze()) + k
            s, r = self.model.step(s, u.unsqueeze(0))
            cost = cost - r \
                        + torch.dot(lam, torch.tanh(mu.squeeze())-ubar) \
                        + (self.rho/2.0) * torch.pow(torch.tanh(mu.squeeze())-ubar,2).sum()
        self.ilqr_optim.zero_grad()
        self.policy_optim.zero_grad()
        cost.backward()
        self.ilqr_optim.step()
        self.policy_optim.step()


        with torch.no_grad():
            K = self.K[0].cpu().clone().numpy()
            k = self.k[0].cpu().clone().numpy()
            xbar = self.xbar[0].cpu().clone().numpy()
            self.k[:-1] = self.k[1:].clone()
            self.k[-1].zero_()
            self.K[:-1] = self.K[1:].clone()
            self.K[-1].zero_()
            self.xbar[:-1] = self.xbar[1:].clone()
            self.xbar[-1].zero_()
            self.lam[:-1] = self.lam[1:].clone()
            self.lam[-1].zero_()
            return np.dot(K, xbar - state) + k
