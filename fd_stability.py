import numpy as np
import matplotlib.pyplot as plt
import math

diff_rate = 0.005
grid_pts = 100
spacing = 1/(grid_pts - 1)
time_step = 0.001
num_steps = 1000
pi = math.pi

pts = np.linspace(0, 1, num = grid_pts)
arr_init = np.sin(pi * pts)

r = diff_rate * time_step/(spacing ** 2)

u = np.copy(arr_init)

for n in range(num_steps):
    u_new = np.zeros_like(u)
    for i in range(1, grid_pts -1):
        u_new[i] = u[i] + r* (u[i+1] - 2*u[i] + u[i-1])
    u = u_new

u_stable = u

time = num_steps * time_step

solution = arr_init * np.exp(-diff_rate * pi **2 * time)

time_step = 0.5
r = diff_rate * time_step/(spacing ** 2)

u = np.copy(arr_init)
for n in range(num_steps):
    u_new = np.zeros_like(u)
    for i in range(1, grid_pts -1):
        u_new[i] = u[i] + r* (u[i+1] - 2*u[i] + u[i-1])
    u = u_new



plt.plot(pts, u_stable, color='b', marker='.', linestyle='none', label="Numerical Solution (r<=0.5)")
plt.plot(pts, solution, color='r', label="Exact Solution")
plt.plot(pts, u, color='o', linestyle='none', label="Numerical Solution (r>0.5)")


plt.xlabel("Position (x)")
plt.ylabel("Temperature (u)")
plt.title("Heat Equation with r <= 0.5 vs r > 0.5")

plt.legend(loc='upper right', fontsize=8)

plt.show()