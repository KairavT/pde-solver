import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

k_rate = 0.5
T_env = 21
T_init = 85

time_pts = torch.linspace(0, 10, 500)
t0= torch.tensor([[0.]])

data = time_pts.reshape(-1,1)
data.requires_grad_(True)


model = nn.Sequential(
    nn.Linear(in_features=1, out_features=32),
    nn.Tanh(),
    nn.Linear(in_features=32, out_features=32),
    nn.Tanh(),
    nn.Linear(in_features=32, out_features=1)
)


optimizer = optim.Adam(model.parameters(), lr = 5e-3)

for i in range(5000):
    temp_T = model(data)
    target_dT = -k_rate * (temp_T - T_env)
    neuralnet_dT = torch.autograd.grad(temp_T, data,
                                 torch.ones_like(temp_T),
                                 create_graph=True)[0]
    
    phys_loss = ((neuralnet_dT - target_dT)**2).mean()
    loss_init = (model(t0) - T_init)**2
    total_loss = phys_loss + loss_init
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()
    if i % 500 == 0: print(total_loss)

nnet_outputs = model(data).detach().squeeze().numpy()
soln_exact = (T_env + (T_init - T_env) * torch.exp(-k_rate * time_pts)).numpy()
x = time_pts.numpy()


plt.plot(x, nnet_outputs, color='b', marker='.', linestyle='none', label="PINN Prediction")
plt.plot(x, soln_exact, color='r', label="Exact Solution")

plt.xlabel("Time (t)")
plt.ylabel("Temperature (T)")
plt.title("PINN vs Exact Solution for Newton's cooling law")

plt.legend()
plt.show()