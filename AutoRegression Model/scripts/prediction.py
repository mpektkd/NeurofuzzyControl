from main import *
import numpy as np
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
#task 3
#With different initial condition, plot the results for 1000 points
x_0 = torch.tensor([-1.9,0,-0.9])
steps = 1001

normalize_input(x_0, maximum,minimum, vector = 'true')
output = torch.zeros((steps,3))
output[0] = x_0.detach().clone()
for i in range (1,steps):    
    output[i] = best_model(output[i-1])

unormalize_output(output,maximum,minimum)

mse = np.zeros(steps-1)

for i in range(steps-1):
    mse[i] = mean_squared_error(X[i].detach().numpy(), output[i].detach().numpy())
    
samples = np.linspace(1, steps, steps)
samples1 = np.linspace(1, steps-1, steps-1)

plt.plot(samples1, mse)
plt.title("MSE between Prediction and Ground Truth")
plt.ylabel("Amplitude")
plt.xlabel("Samples")
plt.show()

x_val = (output.detach().numpy()).T

error = torch.zeros((steps-1,3))
for i in range(steps-1):
    error[i] = X[i] - output[i]

error = torch.transpose(error,0,1)

OUT = (torch.transpose(output, 0,1)).detach().numpy()

fig = plt.figure(figsize=(10,20))
ax = fig.gca(projection='3d')


# Data for a three-dimensional line 
ax.plot3D(OUT[0], OUT[1], OUT[2], 'green', label = '1000 first steps of new Xo')
ax.plot3D(X_old[0,:steps-1].detach().numpy(), X_old[1,:steps-1].detach().numpy(), X_old[2,:steps-1].detach().numpy(), 'red', 
          label = '1000 first steps of old Xo')
ax.legend()
plt.show()

figure, axes = plt.subplots(nrows=1, ncols=3, figsize = (20,5))



axes[0].plot(samples, OUT[0], label = 'prediction for new X0')
axes[0].plot(samples1, X_old[0,:steps-1].detach().numpy(), label = 'real value of old X0')
axes[0].set_title('X-axis')
axes[1].plot(samples, OUT[1], label = 'prediction for new X0')
axes[1].plot(samples1, X_old[1,:steps-1].detach().numpy(), label = 'real value of old X0')
axes[1].set_title('Y-axis')
axes[2].plot(samples, OUT[2], label = 'prediction for new X0')
axes[2].plot(samples1, X_old[2,:steps-1].detach().numpy(), label = 'real value of old X0')
axes[2].set_title('Z-axis')
axes[0].legend()
axes[1].legend()
axes[2].legend()

figure, axes = plt.subplots(nrows=1, ncols=3, figsize = (20,5))
axes[0].plot(samples1, abs(error[0].detach().numpy()))
axes[0].set_title('ΜΑΕ of X-axis')
axes[1].plot(samples1, abs(error[1].detach().numpy()))
axes[1].set_title('ΜΑΕ of Y-axis')
axes[2].plot(samples1, abs(error[2].detach().numpy()))
axes[2].set_title('MAE of Z-axis')
plt.show()

