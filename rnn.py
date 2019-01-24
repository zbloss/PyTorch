import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import numpy as np


class SingleRNN(nn.Module):
    def __init__(self, n_inputs, n_neurons):
        super(SingleRNN, self).__init__()

        self.Wx = torch.randn(n_inputs, n_neurons)  # 4x1
        self.Wy = torch.randn(n_neurons, n_neurons)  # 1x1

        self.b = torch.zeros(1, n_neurons)  # 1x4

    def forward(self, X0, X1):
        self.Y0 = torch.tanh(torch.mm(X0, self.Wx) + self.b)  # 4x1
        self.Y1 = torch.tanh(torch.mm(self.Y0, self.Wy) +
                             torch.mm(X1, self.Wx) + self.b)  # 4x1

        return self.Y0, self.Y1


class BasicRNN(nn.Module):
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    dtype = torch.cuda.FloatTensor

    def __init__(self, n_inputs, n_neurons):
        super(SingleRNN, self).__init__()

        self.Wx = torch.randn(n_inputs, n_neurons).type(
            dtype)  # n_inputs x n_neurons
        self.Wy = torch.randn(n_neurons, n_neurons).type(
            dtype)  # n_neurons x n_neurons

        self.b = torch.zeros(1, n_neurons)  # 1x4

    def forward(self, X0, X1):
        # batch_size x n_neurons
        self.Y0 = torch.tanh(torch.mm(X0, self.Wx) + self.b).cuda()
        self.Y1 = torch.tanh(torch.mm(self.Y0, self.Wy) +
                             torch.mm(X1, self.Wx) + self.b).cuda()  # batch_size x n_neurons

        return self.Y0, self.Y1


N_INPUT = 3  # number of features in input
N_NEURONS = 5  # number of units in layer

X0_batch = torch.tensor([[0, 1, 2], [3, 4, 5],
                         [6, 7, 8], [9, 0, 1]],
                        dtype=torch.float)  # t=0 => 4 X 3

X1_batch = torch.tensor([[9, 8, 7], [0, 0, 0],
                         [6, 5, 4], [3, 2, 1]],
                        dtype=torch.float)  # t=1 => 4 X 3

model = SingleRNN(N_INPUT, N_NEURONS).cuda()

Y0_val, Y1_val = model(X0_batch, X1_batch)

print(f'Y0_val: {Y0_val}')
print(f'Y1_val: {Y1_val}')

print(f'Accomplished using: {torch.cuda.get_device_name(0)}')


# Built-in RNNCell

rnn = nn.RNNCell(3,5) # n_input x n_neurons
X_batch = torch.tensor([[[0,1,2], [3,4,5], 
                         [6,7,8], [9,0,1]],
                        [[9,8,7], [0,0,0], 
                         [6,5,4], [3,2,1]]
                       ], dtype= torch.float) # X0 and X1


hx = torch.randn(4, 5) # m x n_neurons
output = []

# for each time step
for i in range(2):
    hx = rnn(X_batch[i], hx).cuda()
    output.append(hx)

print(output)



class CleanBasicRNN(nn.Module):
    def __init__(self, batch_size, n_inputs, n_neurons):
        super(CleanBasicRNN, self).__init__()

        rnn = nn.RNNCell(n_inputs, n_neurons)
        self.hx = torch.randn(batch_size, n_neurons) # initializing hidden state

    def forward(self, X):
        output = []

        for i in range(2):
            self.hx = rnn(X[i], self.hx)
            output.append(self.hx)

        return output, self.hx

FIXED_BATCH_SIZE = 4 # our batch size is fixed for now
N_INPUT = 3
N_NEURONS = 5

X_batch = torch.tensor([[[0,1,2], [3,4,5], 
                         [6,7,8], [9,0,1]],
                        [[9,8,7], [0,0,0], 
                         [6,5,4], [3,2,1]]
                       ], dtype = torch.float) # X0 and X1


model = CleanBasicRNN(FIXED_BATCH_SIZE, N_INPUT, N_NEURONS).cuda()
output_val, states_val = model(X_batch)
print(f'output_val: {output_val}') # contains all output for all timesteps
print(f'states_val: {states_val}') # contains values for final state or final timestep, i.e., t=1  
