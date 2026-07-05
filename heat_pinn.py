import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np

diff_rate = 0.005
x_min = 0
x_max = 1
t_min = 0
t_max = 1

x_pts = torch.rand(1234,1, requires_grad=True)
t_pts = torch.rand(1234,1, requires_grad=True)

xt_pts = torch.cat([x_pts, t_pts], dim=1)

t0_xvals = torch.linspace(0, 1, 1234).reshape(-1, 1)
t0 = torch.zeros(1234, 1)
t0_pts = torch.cat([t0_xvals, t0], dim=1)

x0 = torch.zeros(1234, 1)
x0_tvals = torch.linspace(0, 1, 1234).reshape(-1, 1)
x0_pts = torch.cat([x0, x0_tvals], dim=1)


x1 = torch.ones(1234, 1)
x1_tvals = torch.linspace(0, 1, 1234).reshape(-1, 1)
x1_pts = torch.cat([x1, x1_tvals], dim=1)

model = nn.Sequential(
    nn.Linear(in_features=2, out_features=32),
    nn.Tanh(),
    nn.Linear(in_features=32, out_features=32),
    nn.Tanh(),
    nn.Linear(in_features=32, out_features=1)
)

optimizer = optim.Adam(model.parameters(), lr = 1e-3)

for i in range(10000):
    temp = model(torch.cat([x_pts, t_pts], dim=1))
    u_t = torch.autograd.grad(temp, t_pts, 
                        grad_outputs=torch.ones_like(temp),
                        create_graph=True)[0]
    u_x = torch.autograd.grad(temp, x_pts, 
                        grad_outputs=torch.ones_like(temp),
                        create_graph=True)[0]
    u_x2 = torch.autograd.grad(u_x, x_pts, 
                        grad_outputs=torch.ones_like(temp),
                        create_graph=True)[0]
    residual = ((u_t - diff_rate * u_x2) **2).mean()

    init_pred = model(t0_pts)
    init_target = torch.sin(t0_pts[:, 0:1] * torch.pi)
    init_loss = ((init_pred - init_target) **2).mean()

    bound0_pred = model(x0_pts)
    bound1_pred = model(x1_pts)
    bound0_loss = ((bound0_pred - torch.zeros_like(bound0_pred)\
                   )**2).mean()
    bound1_loss = ((bound1_pred - torch.zeros_like(bound1_pred)\
                   )**2).mean()

    loss = residual + init_loss + bound0_loss + bound1_loss
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if i % 1000 == 0: print(loss)

def u(x, t):
    return torch.sin(torch.pi * x) * torch.exp(-diff_rate *\
                                                torch.pi **2 *t)
x_axis = torch.linspace(0, 1, 100)
t_axis = torch.linspace(0, 1, 1000)

grid_x, grid_t = torch.meshgrid(x_axis, t_axis, indexing='ij')
grid_x = grid_x.reshape(-1,1)
grid_t = grid_t.reshape(-1,1)

grid_xt = torch.cat([grid_x, grid_t], dim = 1)

with torch.no_grad():
    u_pred = model(grid_xt)

u_actual = u(grid_x, grid_t)

rms_err = ((u_pred - u_actual)**2).mean().sqrt()
max_err = (u_pred - u_actual).abs().max()
print(rms_err, max_err)



# Plot 1: snapshots -- PINN vs exact at fixed times
t_snapshots = [0.0, 0.5, 1.0]
x_plot = torch.linspace(0, 1, 200).reshape(-1, 1)
colors = ['tab:blue', 'tab:green', 'tab:red']

plt.figure(figsize=(9, 6))
for t_snap, c in zip(t_snapshots, colors):
    t_plot = torch.full_like(x_plot, t_snap)
    xt_plot = torch.cat([x_plot, t_plot], dim=1)
    with torch.no_grad():
        pred = model(xt_plot)
    exact = u(x_plot, t_plot)
    plt.plot(x_plot.numpy(), exact.numpy(), color=c, linewidth=2,
             label=f'exact  t={t_snap}')
    plt.plot(x_plot.numpy(), pred.numpy(), color=c, linestyle='dotted', linewidth=2,
             label=f'PINN  t={t_snap}')

plt.xlabel('x')
plt.ylabel('temperature u(x, t)')
plt.title('PINN vs exact solution (1D heat equation)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('snapshots.png', dpi=130)
plt.show()

# Plot 2: error heatmap over the full (x, t) domain
error_grid = (u_pred - u_actual).reshape(len(x_axis), len(t_axis)).numpy()

plt.figure(figsize=(10, 5))
mesh = plt.pcolormesh(t_axis.numpy(), x_axis.numpy(), error_grid,
                      shading='auto', cmap='RdBu')
plt.colorbar(mesh, label='PINN - exact')
plt.xlabel('t')
plt.ylabel('x')
plt.title('Pointwise error over the domain')
plt.tight_layout()
plt.savefig('error_heatmap.png', dpi=130)
plt.show()