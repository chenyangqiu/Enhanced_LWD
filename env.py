import torch
import dgl
from collections import namedtuple
import dgl.function as fn
from copy import deepcopy as dc
import random
import time
from time import time
from torch.utils.data import DataLoader

class MaximumIndependentSetEnv(object):
    def __init__(
        self, 
        max_epi_t, 
        max_num_nodes, 
        hamming_reward_coef, 
        device
    ):
        self.max_epi_t = max_epi_t
        self.max_num_nodes = max_num_nodes
        self.hamming_reward_coef = hamming_reward_coef
        self.device = device

    def step(self, action):
        reward, sol, done = self._take_action(action)
        ob = self._build_ob()
        self.sol = sol
        info = {"sol": self.sol}
        return ob, reward, done, info

    # compute legality penalty
    def compute_legality_penalty(self, k=10.0):
        """Soft legality penalty: penalize adjacent nodes both selected (x_i = x_j = 1)"""
        # Select probability-like activation via sigmoid
        x1 = (self.x == 1).float()  # shape: [num_nodes, num_samples]
        x1_sigmoid = torch.sigmoid(k * x1)

        # Message passing: each node receives sum of selected neighbors
        self.g.ndata['h'] = x1_sigmoid
        self.g.update_all(fn.copy_u('h', 'm'), fn.sum('m', 'h'))
        neighbor_sum = self.g.ndata.pop('h')  # [num_nodes, num_samples]

        # Compute conflicts: if x_i and neighbors both selected
        penalty = (x1_sigmoid * neighbor_sum).sum(dim=0)  # sum over nodes → [num_samples]
        return penalty.mean()  # return scalar penalty

    def _take_action(self, action):
        undecided = self.x == 2
        self.x[undecided] = action[undecided]
        self.t += 1

        x1 = (self.x == 1)
        self.g = self.g.to(self.device)
        self.g.ndata['h'] = x1.float()
        self.g.update_all(
            fn.copy_u('h', 'm'),
            fn.sum('m', 'h')
        )
        x1_deg = self.g.ndata.pop('h')

        # forgive clashing
        clashed = x1 & (x1_deg > 0)
        self.x[clashed] = 2
        x1_deg[clashed] = 0

        still_undecided = (self.x == 2)
        self.x[still_undecided & (x1_deg > 0)] = 0

        still_undecided = (self.x == 2)
        timeout = (self.t == self.max_epi_t)
        self.x[still_undecided & timeout] = 0

        done = self._check_done()
        # self.epi_t[~done] += 1
        self.epi_t[:, ~done] += 1

        x1 = (self.x == 1).float()
        node_sol = x1

        self.g.ndata['h'] = node_sol
        next_sol = self.g.ndata['h'].sum(dim=0)  # replacement of dgl.sum_nodes
        self.g.ndata.pop('h')

        reward = (next_sol - self.sol)

        if self.hamming_reward_coef > 0.0 and self.num_samples == 2:
            xl, xr = self.x.split(1, dim=1)
            undecidedl, undecidedr = undecided.split(1, dim=1)
            hamming_d = torch.abs(xl.float() - xr.float())
            hamming_d[(xl == 2) | (xr == 2)] = 0.0
            hamming_d[~undecidedl & ~undecidedr] = 0.0
            self.g.ndata['h'] = hamming_d
            hamming_reward = self.g.ndata['h'].sum(dim=0).expand_as(reward)  # ✅ 替代 dgl.sum_nodes
            self.g.ndata.pop('h')
            reward += self.hamming_reward_coef * hamming_reward

        reward /= self.max_num_nodes
        return reward, next_sol, done

    def _check_done(self):
        undecided = (self.x == 2).float()
        self.g.ndata['h'] = undecided
        num_undecided = self.g.ndata['h'].sum(dim=0)  # ✅ 替代 dgl.sum_nodes
        self.g.ndata.pop('h')
        done = (num_undecided == 0)
        return done

    def _build_ob(self):
        ob_x = self.x.unsqueeze(2).float()
        ob_t = self.t.unsqueeze(2).float() / self.max_epi_t
        ob = torch.cat([ob_x, ob_t], dim=2)
        return ob

    def register(self, g, num_samples=1):
        self.g = g
        self.num_samples = num_samples
        # self.g.set_n_initializer(dgl.init.zero_initializer)  # ✅ 可省略
        self.g = self.g.to(self.device)
        # self.batch_num_nodes = torch.LongTensor(self.g.batch_num_nodes()).to(self.device)
        self.batch_num_nodes = self.g.batch_num_nodes().to(self.device)

        num_nodes = self.g.number_of_nodes()
        self.x = torch.full(
            (num_nodes, num_samples), 2,
            dtype=torch.long,
            device=self.device
        )
        self.t = torch.zeros(
            num_nodes, num_samples,
            dtype=torch.long,
            device=self.device
        )

        ob = self._build_ob()

        self.sol = torch.zeros(
            self.g.batch_size,
            num_samples,
            device=self.device
        )
        self.epi_t = torch.zeros(
            self.g.batch_size,
            num_samples,
            device=self.device
        )
        return ob
