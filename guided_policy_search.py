import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal

class GuidedPolicySearch(object):

    def __init__(self, model, policy, T=10, lr=0.1):
        self.T = T
        self.model = model
        self.policy= policy

        self.state_dim = model.num_states
        self.action_dim = model.num_actions

        self.device = 'cpu'
        if torch.cuda.is_available():
            self.device = 'cuda:0'
        self.lr = lr
        self.u = torch.zeros(T, self.action_dim).to(self.device)
        self.lam = torch.zeros(T, self.action_dim).to(self.device)

        self.u.requires_grad = True
        self.rho = 1e-1
        # self.u_optim = optim.Adam(, lr=3e-3)
        # self.lam_optim = optim.SGD([self.lam], lr=3e-3)
        self.optim = optim.Adam(self.policy.parameters(), lr=3e-4)
    def reset(self):
        with torch.no_grad():
            self.u.zero_()
            self.lam.zero_()

    def update_constraint(self, state):
        with torch.no_grad():
            s = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            cost = 0.
            i = 0
            for u, lam in zip(self.u, self.lam):
                mu, _ = self.policy(s)
                s, r = self.model.step(s, u.unsqueeze(0))
                self.lam[i] += self.rho * (torch.tanh(mu.squeeze()) - u)
                i += 1
    def update(self, state):
        self.update_constraint(state)
        s = torch.FloatTensor(state).unsqueeze(0).to(self.device)

        cost = 0.
        rho_2 = self.rho/2.0
        for u, lam in zip(self.u, self.lam.detach()):
            mu, _ = self.policy(s)
            s, r = self.model.step(s, u.unsqueeze(0))
            cost = cost - r \
                        + torch.dot(lam, torch.tanh(mu.squeeze())-u) \
                        + rho_2 * torch.pow(torch.tanh(mu.squeeze())-u,2).sum()

        self.optim.zero_grad()
        cost.backward()
        self.optim.step()
        print(cost.item())
        with torch.no_grad():
            self.u -= self.lr * self.u.grad
            self.u.grad.zero_()
            u = self.u[0].cpu().clone().numpy()
            self.u[:-1] = self.u[1:].clone()
            self.u[-1].zero_()
            self.lam[:-1] = self.lam[1:].clone()
            self.lam[-1].zero_()
        return u
