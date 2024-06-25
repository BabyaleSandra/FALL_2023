from env.cartpole import CartPole
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import control as ct

class LinearQuadraticRegulator(nn.Module):
    def __init__(self, device='cpu'):
        super(LinearQuadraticRegulator, self).__init__()
        mc = 0.2
        mp = 0.5
        l = 0.5
        g = 9.80364
        A = np.array(
            [[0, 0, 1, 0], [0, 0, 0, 1], [0, mp*g/mc, 0, 0], [0, (mc+mp)*g/mc/l, 0, 0]]
        )
        B = np.array(
            [[0], [0], [1/mc], [1/mc/l]]
        )
        Q = np.diag([1/20, mp*g*l, 1/2*(mc+mp), 1/2*mp*l])
        Q[2,3] = 1/2*mp*l
        Q[3,2] = Q[2,3]
        R = 1/10*np.eye(1)
        K, S, E = ct.lqr(A, B, Q, R)
        self.K = torch.tensor(K, dtype=torch.float).to(device)

    def forward(self, x):
        return -self.K @ x

class Controller(nn.Module):
    """ if hiddenlayers=[] and bias=False, this yields a linear controller. """
    def __init__(self, hiddenlayers=[15], activation=nn.SiLU(), bias=True, p=0.0):
        super(Controller, self).__init__()
        self.fcl = nn.ModuleList()
        self.dropout = nn.Dropout(p=p)
        
        if len(hiddenlayers) > 0:
            self.fcl.append(nn.Linear(4, hiddenlayers[0], bias=bias))
            torch.nn.init.xavier_normal_(self.fcl[-1].weight)
            self.fcl.append(activation)
            for i in range(len(hiddenlayers)-1):
                self.fcl.append(nn.Linear(hiddenlayers[i], hiddenlayers[i+1], 
                                          bias=bias))
                torch.nn.init.xavier_normal_(self.fcl[-1].weight)
                self.fcl.append(activation)
            self.fcl.append(nn.Linear(hiddenlayers[-1], 1, bias=False))
        else:
            self.fcl.append(nn.Linear(4, 1, bias=bias))
        torch.nn.init.xavier_normal_(self.fcl[-1].weight)

    def forward(self, x):
        for i, l in enumerate(self.fcl):
            if i == 0:
                x = l(x)
            else:
                x = l(x)
                x = self.dropout(x)
        return x

class TrajectoryLoss(nn.Module):
    def __init__(self):
        super(TrajectoryLoss, self).__init__()
        self.mc, self.mp = 0.2, 0.5
        self.l, self.g = 0.5, 9.80364

    def forward(self, inputs, ctrl):
        N = len(inputs)
        loss = 0
        for i in range(N):
            loss = loss + 0.05*inputs[i,0]**2 + \
                self.mp*self.g*self.l*(1-torch.cos(inputs[i,1])) + \
                1/2*(self.mc+self.mp)*inputs[i,2]**2 + \
                self.mp*self.l*inputs[i,2]*inputs[i,3]*torch.cos(inputs[i,1]) + \
                1/2*self.mp*self.l*self.l*inputs[i,3]**2 + 1/10*ctrl[i,0]**2
        return loss/N

class ControlSystem(nn.Module):
    def __init__(self, c=Controller([], bias=False), device='cpu'):
        super(ControlSystem, self).__init__()
        self.robot = CartPole(device)
        self.controller = c
        self.loss_fn = TrajectoryLoss()
        if len(self.controller.fcl) < 2:
            self.tf, self.n = 20.0, 200
        else:
            self.tf, self.n = 2.0, 32

    def forward(self, x):
        N = len(x)
        t, c = self.robot.simulate(controller=self.controller, 
                                s0=x[0], tf=self.tf, n=self.n)
        l = torch.reshape(self.loss_fn(t, c), (1,1))
        del t, c
        for i in range(1,N):
            t, c = self.robot.simulate(controller=self.controller, 
                                    s0=x[i], tf=self.tf, n=self.n)
            l = torch.cat((l, torch.reshape(self.loss_fn(t, c), (1,1))), dim=0)
            del t, c
        return l