from numpy import floor, ceil
import torch
from copy import deepcopy

class CartPole(object):
    def __init__(self, device):
        """ Model is taken from:
            https://underactuated.mit.edu/acrobot.html#cart_pole
            Our theta = their theta + pi
        """
        self.device = device
        self.delta_t = 0.01
        self.num_states = 4
        self.num_inputs = 1
        self.cart_mass = 0.2 # kg
        self.pendulum_mass = 0.5 # kg
        # https://geodesy.noaa.gov/cgi-bin/grav_pdx.prl (Boise State Engineering)
        # (lat, long) = (53.5885, -116.1918 E)
        # elevation = 2723ft or 830m
        self.gravitational_acc = 9.80364 # m/s/s
        self.pendulum_length = 0.5 # m
        self.state = torch.zeros(self.num_states).to(self.device) # (x, theta, xdot, thetadot)
        self.control = torch.zeros(self.num_inputs).to(self.device) # force applied on the cart
        self.control_limit = 150.0
        self._compute_mass_matrix() # mass matrix
        self._compute_coriolis_matrix() # coriolis and centripetal matrix
        self._compute_gravitational_torque() # graviatational torque
        self.B = torch.tensor([1, 0]).to(self.device) # input matrix

    def _compute_mass_matrix(self):
        q = self.get_position()
        M = torch.zeros((self.num_states//2,self.num_states//2)).to(self.device)
        M[0,0] = self.cart_mass + self.pendulum_mass
        M[1,1] = self.pendulum_mass * self.pendulum_length**2
        M[0,1] = -self.pendulum_mass * self.pendulum_length * \
                    torch.cos(q[1])
        M[1,0] = M[0,1]
        self.M = M

    def _compute_coriolis_matrix(self):
        q = self.get_position()
        qdot = self.get_velocity()
        C = torch.zeros((self.num_states//2,self.num_states//2)).to(self.device)
        C[0,1] = self.pendulum_mass * self.pendulum_length * \
                    torch.sin(q[1]) * qdot[1]
        self.C = C

    def _compute_gravitational_torque(self):
        q = self.get_position()
        G = torch.zeros(self.num_states//2).to(self.device)
        G[1] = self.pendulum_mass * self.gravitational_acc * \
                self.pendulum_length * torch.sin(q[1])
        self.G = G

    def get_position(self):
        return self.state[0:self.num_states//2].clone()

    def get_velocity(self):    
        return self.state[self.num_states//2:self.num_states].clone()

    # def set_position(self, s):
    #     for i in range(2):
    #         self.state[i] = s[i].clone()
    #     self._compute_mass_matrix()
    #     self._compute_coriolis_matrix()
    #     self._compute_gravitational_torque()

    # def set_velocity(self, s):
    #     for i in range(2,4):
    #         self.state[i] = s[i].clone()
    #     self._compute_coriolis_matrix()

    def set_state(self, s):
        self.state = s
        self._compute_mass_matrix()
        self._compute_coriolis_matrix()
        self._compute_gravitational_torque()

    def set_control(self, tau):
        self.control = torch.clamp(tau, -self.control_limit, self.control_limit)

    def reset(self):
        self.set_state(torch.zeros(self.num_states))
    
    def step(self):
        q = self.get_position()
        qdot = self.get_velocity()
        rhs1 = q + qdot * self.delta_t
        rhs2 = qdot + torch.linalg.solve(self.M,
                self.G - self.C @ qdot + self.B * self.control)*self.delta_t
        return torch.cat((rhs1, rhs2), dim=0)

    def simulate(self, 
                 controller = lambda s: -s[0]-s[2], 
                 s0=torch.tensor([0., torch.pi/3, 0, 0]), 
                 tf=3.0, n=10):
        self.set_state(torch.autograd.Variable(s0, requires_grad=True))
        self.set_control(controller(self.state))
        trajectory = torch.empty((0, self.num_states)).to(self.device)
        trajectory = torch.cat((trajectory, 
                    torch.reshape(self.state, (1, self.num_states))), dim=0)
        ctrl = torch.empty((0, self.num_inputs)).to(self.device)
        ctrl = torch.cat((ctrl, 
                    torch.reshape(self.control, (1, self.num_inputs))), dim=0)
        N = ceil(1+tf/self.delta_t).astype(int)
        m = max(1, floor(N/n).astype(int))
        for i in range(N):
            self.set_control(controller(self.state))
            self.set_state(self.step())
            if i % m == 0:
                trajectory = torch.cat((trajectory, 
                    torch.reshape(self.state, (1, self.num_states))), dim=0)
                ctrl = torch.cat((ctrl, 
                    torch.reshape(self.control, (1, self.num_inputs))), dim=0)
            if 1-torch.cos(self.state[1]) > 0.2:
                break
        self.reset()
        return trajectory, ctrl

    def random_state(self):
        return torch.mul(torch.rand(self.num_states), 
                         torch.tensor([0.0, 0, 0, 0.5])) - \
                         torch.tensor([0.0, 0, 0, 0.25])