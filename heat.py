
import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

class NN(nn.Module):

    def __init__(self):
        super(NN, self).__init__()
        self.seq = nn.Sequential(
            nn.Linear(2, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
            nn.Tanh()
        )

    def forward(self, x, t):

        data = torch.cat([x, t], dim = 1)
        return self.seq(data)

class PINN:

    def __init__(self, x, t, ic_func, bc0, bc1):

        self.x_initial = x
        self.t_initial = t

        self.linear_points = 1000

        symbols = sp.symbols('x')
        expr = sp.sympify(ic_func)
        self.ic0 = sp.lambdify(symbols, expr, modules=['numpy'])

        self.bc0 = bc0
        self.bc1 = bc1

        self.alpha = 0.1
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.model = NN()
        self.model.to(self.device)
        self.generate_values()
        
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr = 0.001)

    def generate_values(self):
        
        self.x_ic = torch.linspace(0, self.x_initial, self.linear_points).view(-1, 1).to(self.device)
        self.t_ic = torch.zeros_like(self.x_ic).to(self.device)
        self.u_ic = torch.sin(torch.pi * self.x_ic).to(self.device)
        self.u_ic = torch.tensor(self.ic0(self.x_ic)).to(self.device)

        self.t_bc = torch.linspace(0, self.t_initial, self.linear_points).view(-1, 1).to(self.device)
        self.x_bc0 = torch.zeros_like(self.t_bc).to(self.device)
        self.u_bc0 = torch.zeros_like(self.t_bc).to(self.device)
        self.x_bc1 = torch.ones_like(self.t_bc).to(self.device)
        self.u_bc1 = torch.zeros_like(self.t_bc).to(self.device)

        self.x = torch.rand(1000).view(-1, 1).to(self.device)
        self.x.requires_grad = True
        self.t = torch.rand(1000).view(-1, 1).to(self.device)
        self.t.requires_grad = True


    
    def calc_pde_loss(self):

        u = self.model(self.x, self.t)

        u_x = torch.autograd.grad(u, self.x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        u_t = torch.autograd.grad(u, self.t, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        u_xx = torch.autograd.grad(u_x, self.x, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]

        return torch.mean((u_t - self.alpha * u_xx) ** 2)


    def train(self, epochs = 1000):
        self.model.train()
        self.epoch_loss = []
        self.ic_loss = []
        self.bc_loss = []
        self.pde_loss = []
        for epoch in range(epochs):
            self.optimizer.zero_grad()

            u_ic_pred = self.model(self.x_ic, self.t_ic)
            loss_ic = torch.mean((self.u_ic - u_ic_pred) ** 2)

            u_bc0_pred = self.model(self.x_bc0, self.t_bc)
            u_bc1_pred = self.model(self.x_bc1, self.t_bc)
            loss_bc = torch.mean((self.u_bc0 - u_bc0_pred) ** 2) + torch.mean((self.u_bc1 - u_bc1_pred) ** 2)

            loss_pde = self.calc_pde_loss()

            loss = loss_ic + loss_bc + loss_pde
            
            self.ic_loss.append(loss_ic.item())
            self.bc_loss.append(loss_bc.item())
            self.pde_loss.append(loss_pde.item())
            self.epoch_loss.append(loss.item())

            loss.backward()
            self.optimizer.step()
            if epoch%100 == 0:
                print(f"Epoch : {epoch}\nloss_ic : {loss_ic}\nloss_bc : {loss_bc}\npde_loss : {loss_pde}")

    def predict(self, x, t):
        self.model.eval()
        with torch.no_grad():
            predicted = self.model(torch.tensor(x).view(-1, 1), torch.tensor(t).view(-1, 1))
        return predicted
    
    def inference(self):
        self.x_test = torch.linspace(0, self.x_initial, self.linear_points).cpu()
        self.t_test = torch.linspace(0, self.t_initial, self.linear_points).cpu()

        self.x_test, self.t_test = torch.meshgrid(self.x_test,  self.t_test)

        self.model.eval()
        self.model.cpu()
        with torch.no_grad():
            self.output = self.model(self.x_test.reshape(-1, 1), self.t_test.reshape(-1, 1))
        # return self.x_test, self.t_test, output.reshape(self.linear_points, self.linear_points)
    
    def plot_test_3d(self):
        self.inference()
        fig = plt.figure()
        ax = plt.axes(projection= '3d')
        # print(self.x_test.shape, self.t_test.shape, self.output.shape)
        ax.plot_surface(self.x_test, self.t_test, self.output.reshape(self.linear_points, self.linear_points), cmap = 'viridis')
        ax.set_xlabel("Length")
        ax.set_ylabel("Time")
        ax.set_zlabel("Temperature")
        return fig


    def plot_loss(self):
        fig = plt.figure(figsize = (12, 6))
        plt.plot(range(len(self.ic_loss)), self.ic_loss,label = "Initial Condition")
        plt.plot(range(len(self.bc_loss)), self.bc_loss, label = "Boundry Condition")
        plt.plot(range(len(self.pde_loss)), self.pde_loss, label = "PDE Loss")
        plt.plot(range(len(self.epoch_loss)), self.epoch_loss, label = "Total Loss")
        plt.legend()
        return fig

