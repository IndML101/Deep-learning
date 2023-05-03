import numpy as np
from collections import defaultdict
from matplotlib import pyplot as plt
import json

class NeuralNet:
    def __init__(self, X, y, layers: tuple, batch_size=32, loss: str = 'cross-entropy') -> None:
        self.X = X
        self.y = y
        self.loss = loss
        self.layers = layers
        self.batch_size = batch_size
        self.network = defaultdict(dict)
        self.logs = defaultdict(dict)
        assert self.X.shape[1] == self.layers[0]

    def init_weights(self):
        n_layers = len(self.layers) - 1
        for i in range(1, n_layers):
            key = 'hidden_' + str(i)
            self.network[key]['weights'] = np.random.standard_normal((self.layers[i], self.layers[i-1]))
            self.network[key]['biases'] = np.random.standard_normal((self.layers[i], 1))
            self.network[key]['activation'] = 'Relu'


        key = 'hidden_' + str(n_layers)
        self.network[key]['weights'] = np.random.standard_normal((self.layers[n_layers], self.layers[n_layers-1]))
        self.network[key]['biases'] = np.random.standard_normal((self.layers[n_layers], 1))
        self.network[key]['activation'] = 'Softmax'


    def relu(self, arr):
        return np.maximum(0, arr)
    
    def softmax(self, arr):
        arr = arr - arr.max(axis=0)
        return np.exp(arr)/np.exp(arr).sum(axis=0)
    
    def cross_entropy(self, y, yhat):
        epsilon = 1e-9
        return -np.mean(y*np.log(yhat+epsilon))
    
    def mse(self, y, yhat):
        return np.mean(np.square(yhat-y))
    
    def gradient_relu(self, x):
        return np.where(x>0, x, 0)

    def gradient_output(self, y):
        output_index = len(self.layers)-1
        key = 'hidden_'+str(output_index)

        if self.loss == 'cross-entropy':
            return y.transpose()-self.network[key]['hidden_activation']
        
        return (self.network[key]['hidden_activation'] - y.transpose()) * 2


    def forward(self, X):
        n_layers = len(self.layers)

        key = 'hidden_'+str(1)
        self.network[key]['hidden_state'] = np.dot(self.network[key]['weights'], X.transpose()) + np.repeat(self.network[key]['biases'], X.shape[0], axis=1)
        if self.network[key]['activation'] == 'Relu':
            self.network[key]['hidden_activation'] = self.relu(self.network[key]['hidden_state'])
        else:
            self.network[key]['hidden_activation'] = self.softmax(self.network[key]['hidden_state'])

        for i in range(2, n_layers):
            keyi = 'hidden_'+str(i)
            key_prev = 'hidden_'+str(i-1)
            self.network[keyi]['hidden_state'] = np.dot(self.network[keyi]['weights'],self.network[key_prev]['hidden_activation']) + np.repeat(self.network[keyi]['biases'], self.network[key_prev]['hidden_activation'].shape[1], axis=1)
            if self.network[keyi]['activation'] == 'Relu':
                self.network[keyi]['hidden_activation'] = self.relu(self.network[keyi]['hidden_state'])
            else:
                self.network[keyi]['hidden_activation'] = self.softmax(self.network[keyi]['hidden_state'])

        key = 'hidden_'+str(n_layers-1)
        return self.network[key]['hidden_activation']

    def backprop(self, X, y, lr):
        # gradient wrpt output layer
        n_layers = len(self.layers)-1
        gradient_al = self.gradient_output(y)

        for i in range(n_layers, 1, -1):
            key = 'hidden_'+str(i)
            key_prev = 'hidden_'+str(i-1)
            self.network[key]['gradient_w'] = np.dot(gradient_al, self.network[key_prev]['hidden_activation'].transpose())
            self.network[key]['gradient_b'] = gradient_al.sum(axis=1).reshape((self.network[key]['biases'].shape))

            gradient_hi = np.dot(self.network[key]['weights'].transpose(), gradient_al)
            gradient_al = gradient_hi*self.gradient_relu(self.network[key_prev]['hidden_activation'])

            self.network[key]['weights'] -= lr*self.network[key]['gradient_w']
            self.network[key]['biases'] -= lr*self.network[key]['gradient_b']


        key = 'hidden_'+str(1)
        self.network[key]['gradient_w'] = np.dot(gradient_al, X)
        self.network[key]['gradient_b'] = gradient_al.sum(axis=1).reshape(self.network[key]['biases'].shape)

        self.network[key]['weights'] -= lr*self.network[key]['gradient_w']
        self.network[key]['biases'] -= lr*self.network[key]['gradient_b']


    def optimize(self, n_epochs=100, lr=1e-3):
        self.init_weights()
        for i in range(n_epochs):
            for j in range(0, self.X.shape[0], self.batch_size):
                if j+self.batch_size > self.X.shape[0]:
                    X = self.X[j:,:]
                    y = self.y[j:,:]
                else:
                    X = self.X[j:j+self.batch_size,:]
                    y = self.y[j:j+self.batch_size,:]
                yhat = self.forward(X)
                self.backprop(X, y, lr)

                yield self.cross_entropy(y, yhat.transpose())

            net_dict = defaultdict(dict)
            for key in self.network:
                for k, v in self.network[key].items():
                    if isinstance(v, np.ndarray):
                        net_dict[key][k] = v.tolist()

            self.logs['epoch'+str(i)] = net_dict


if __name__ == '__main__':
    n_samples = 500
    X = np.random.standard_normal((n_samples,5))
    y = np.random.randint(0,2, size=(n_samples, 1))
    y = np.concatenate((y, ~y), axis=1).reshape((n_samples,2))
    layers = (X.shape[1], 16, 16, 2)

    ann = NeuralNet(X=X, y=y, layers=layers)
    training_loss = ann.optimize(n_epochs=200, lr=1e-6)
    training_loss = list(training_loss)

    with open('../artifacts/backprop-logs.json', 'w+') as f:
        json.dump(ann.logs, f)

    fig, axs = plt.subplots(figsize=(10,7))
    plt.plot(list(range(len(training_loss))), training_loss)
    plt.title("Backprop loss")
    plt.ylabel('Loss')
    plt.xlabel('Epochs')
    plt.savefig('../artifacts/backprop-loss.png')