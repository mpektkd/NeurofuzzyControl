from main import *
import numpy as np
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

#task 2
#Execute the prediction of the model for 1000 points with same initial condition and
#plot the errors, that converge to zero
prediction = torch.zeros((1000,3))
err = torch.zeros((1000,3))
for i in range (1000):
    prediction[i] = best_model(X[i])
    unormalize_output(prediction[i], maximum,minimum, vector = 'true')
    err[i] = Y[i] - prediction[i]

ERROR = torch.transpose(err,0,1)
e = ERROR.detach().numpy()
pred = (torch.transpose(prediction,0,1)).detach().numpy()

samples = np.linspace(1, 1000, 1000)
figure, axes = plt.subplots(nrows=1, ncols=3, figsize = (20,5))
plt.figure(figsize=(10,20))

axes[0].plot(samples, abs(e[0]), 'tab:red')
axes[0].set_title('MAE of X-axis')
axes[1].plot(samples, abs(e[1]), 'tab:green')
axes[1].set_title('MAE of Y-axis')
axes[2].plot(samples, abs(e[2]), 'tab:orange')
axes[2].set_title('MAE of Z-axis')

plt.show()

figure, axes = plt.subplots(nrows=1, ncols=3, figsize = (20,5))

axes[0].plot(samples, pred[0], label = 'prediction')
axes[0].plot(samples, Y_old[0], label = 'real value')
axes[0].set_title('X-axis')
axes[1].plot(samples, pred[1], label = 'prediction')
axes[1].plot(samples, Y_old[1], label = 'real value')
axes[1].set_title('Y-axis')
axes[2].plot(samples, pred[2], label = 'prediction')
axes[2].plot(samples, Y_old[2], label = 'real value')
axes[2].set_title('Z-axis')
axes[0].legend()
axes[1].legend()
axes[2].legend()
plt.show()

for ax in axes.flat:
    ax.set(xlabel='Samples', ylabel='Amplitude')

figure.tight_layout()

mse = np.zeros(1000)

for i in range(1000):
    mse[i] = mean_squared_error(Y[i].detach().numpy(), prediction[i].detach().numpy())
    
plt.plot(samples, mse)
plt.title("MSE between Prediction and Ground Truth")
plt.ylabel("Amplitude")
plt.xlabel("Samples")
plt.show()