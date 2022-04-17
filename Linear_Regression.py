# 1) Designe model
# 2) Construct loss and optimizer
# 3) Training loop
#   - forward pass : compute prediction and loss
#   - backward pass : gradient
#   - update weights

import torch
import torch.nn as nn
import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt

# 0) prepare data
X_numpy, Y_numpy = datasets.make_regression(n_samples=100, n_features=1, noise=20, random_state=1)
X = torch.from_numpy(X_numpy.astype(np.float32))
y = torch.from_numpy(Y_numpy.astype(np.float32))
y = y.view(y.shape[0], 1)

n_samples, n_feature = X.shape

# 1) model 
input_size = n_feature # 1
output_size = 1 # 1
model = nn.Linear(n_feature, output_size)

# 2) loss and optimizer
learning_rate = 0.01
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# 3) training loop
num_epocs = 1
for epoc in range(num_epocs):
    # forward pass and loss
    y_predicted = model(X)
    loss = criterion(y_predicted, y)
    print(loss)

    # backpass
    loss.backword()    

    # update
    optimizer.step()

    #IMPORTANT
    optimizer.zero_grad()

    if (epoc+1) % 10 == 0:
        print(f'epoc: {epoc+1}, loss = {loss.item():.4f}')

# plot
predicted = model(X).detach() # new tensor, where grad = False
plt.plot(X_numpy, Y_numpy, 'ro')
plt.plot(X_numpy, predicted, 'b')
plt.show()
