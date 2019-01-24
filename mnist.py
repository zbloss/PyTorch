import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import numpy as np
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim

if __name__ == '__main__':

    BATCH_SIZE = 64

    # list all transformations
    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    # download and load the training dataset

    trainset = torchvision.datasets.MNIST(
        root='./data', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

    # download and load the testing dataset

    testset = torchvision.datasets.MNIST(
        root='./data', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

    # params
    N_STEPS = 28
    N_INPUTS = 28
    N_NEURONS = 150
    N_OUTPUTS = 10
    N_EPOCHS = 10

    class ImageRNN(nn.Module):
        def __init__(self, batch_size, n_steps, n_inputs, n_neurons, n_outputs):
            super(ImageRNN, self).__init__()

            self.n_neurons = n_neurons
            self.batch_size = batch_size
            self.n_steps = n_steps
            self.n_inputs = n_inputs
            self.n_outputs = n_outputs

            self.basic_rnn = nn.RNN(self.n_inputs, self.n_neurons)

            self.FC = nn.Linear(self.n_neurons, self.n_outputs)

        def init_hidden(self, ):
            # (num_layers, batch_size, n_neurons)
            return (torch.zeros(1, self.batch_size, self.n_neurons)).to(device)

        def forward(self, X):
            # transforms X to dimensions: n_steps x batch_size x n_inputs
            X = X.permute(1, 0, 2)

            self.batch_size = X.size(1)
            self.hidden = self.init_hidden()

            lstm_out, self.hidden = self.basic_rnn(X, self.hidden)
            out = self.FC(self.hidden)

            return out.view(-1, self.n_outputs)  # batch_size x n_output

    # testing the model
    #dataiter = iter(trainloader)
    #images, labels = dataiter.next()
    # model = ImageRNN(BATCH_SIZE, N_STEPS, N_INPUTS, N_NEURONS, N_OUTPUTS)  #.cuda()
    #logits = model(images.view(-1, 28, 28))
    # print(logits[0:10])

    # Setting Device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Model Instance
    model = ImageRNN(BATCH_SIZE, N_STEPS, N_INPUTS,
                     N_NEURONS, N_OUTPUTS)
    model = model.to(device)
    print(f'Device: {device}')

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    def get_accuracy(logit, target, batch_size):
        # training accuracy
        corrects = (torch.max(logit, 1)[1].view(
            target.size()).data == target.data).sum()
        accuracy = 100.0 * corrects / batch_size
        return accuracy.item()

    for epoch in range(N_EPOCHS):
        train_running_loss = 0.0
        train_acc = 0.0
        model.train()

        # Training Round
        for i, data in enumerate(trainloader):
            # zero the parameter gradients
            optimizer.zero_grad()

            # reset hidden states
            model.hidden = model.init_hidden()

            # get the inputs
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            inputs = inputs.view(-1, 28, 28)

            # forward + backward + optimize
            outputs = model(inputs)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_running_loss += loss.detach().item()
            train_acc += get_accuracy(outputs, labels, BATCH_SIZE)

        model.eval()
        test_acc = 0.0
        for i, data in enumerate(testloader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            inputs = inputs.view(-1, 28, 28)
            outputs = model(inputs)
            test_acc += get_accuracy(outputs, labels, BATCH_SIZE)

        print(
            f'EPOCH: {epoch}; Loss: {train_running_loss / i}; Train Accuracy: {train_acc / i}')
        print(f'Test Accuracy: {test_acc / i}')
