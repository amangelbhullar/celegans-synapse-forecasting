import math, torch, torch.nn as nn, torch.nn.functional as F
class SparsityController(nn.Module):
    def __init__(self, dim, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3 * dim, hidden), nn.ReLU(),
            nn.Linear(hidden, 1)
        )

    def forward(self, eh, er, et, tau=0.5, hard=True, train_mode=True):
        s = self.net(torch.cat([eh, er, et], dim=-1)).squeeze(-1)
        # Hard-concrete gate (Louizos et al., 2018): stretched [gamma, zeta] + clamp.
        # Produces exact zeros; prob below is the closed-form P(gate != 0),
        # so `lam * prob.mean()` in the training loop is the proper L0 penalty.
        beta, gamma, zeta = 2.0 / 3.0, -0.1, 1.1
        if train_mode:
            u = torch.rand_like(s).clamp(1e-6, 1 - 1e-6)
            z = torch.sigmoid((torch.log(u) - torch.log(1 - u) + s) / beta)
        else:
            z = torch.sigmoid(s)
        mask = torch.clamp(z * (zeta - gamma) + gamma, 0.0, 1.0)
        prob = torch.sigmoid(s - beta * math.log(-gamma / zeta))
        return mask, prob


