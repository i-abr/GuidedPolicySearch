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
        self.lam = torch.ones(T, self.action_dim).to(self.device)

        self.u.requires_grad = True
        self.lam.requires_grad = True

        # self.u_optim = optim.Adam(, lr=3e-3)
        # self.lam_optim = optim.SGD([self.lam], lr=3e-3)
    def reset(self):
        with torch.no_grad():
            self.u.zero_()

    def __call__(self, state):
        s = torch.FloatTensor(state).unsqueeze(0).to(self.device)

        cost = 0.
        for u, lam in zip(self.u, self.lam):
            mu, _ = self.policy(s)
            s, r = self.model.step(s, u.unsqueeze(0))
            cost = cost - r #+ torch.dot(lam, u-mu)

        cost.backward()
        with torch.no_grad():
            self.u -= self.lr * self.u.grad
            self.u.grad.zero_()
            u = self.u[0].cpu().clone().numpy()
            self.u[:-1] = self.u[1:].clone()
            self.u[-1].zero_()
            return u
