import numpy as np
import matplotlib.pyplot as plt



class PerceptronLEarning:
    def __init__(self, X, y, W, b) -> None:
        self.X = X
        self.y = y
        self.W = W
        self.b = b
        self.mask = np.ones((X.shape[0], 1))
        self.idx = 0

    def linear(self):
        return np.matmul(self.X, self.W) + self.b*self.mask

    def convergence(self):
        fx = self.linear()
        idx = self.y.shape[0]//2
        loss0 = (self.y *fx)[:idx,:] > 0
        loss1 = ((1-self.y) *fx)[idx:, :] < 0

        converge = loss0.all() and loss1.all()
        return converge, -np.average((loss1.sum(), loss0.sum()))

    def train(self):
        converge = False
        while not converge: 
            converge, loss = self.convergence()
            self.idx = np.random.randint(self.y.shape[0])                

            if self.y[self.idx] == 1 and np.dot(self.W.transpose().flatten(), self.X[self.idx,:].flatten()) + self.b < 0:
                self.W += self.X[self.idx,:].reshape(self.W.shape)
                self.b += 1
            
            if self.y[self.idx] == 0 and np.dot(self.W.transpose().flatten(), self.X[self.idx,:].flatten()) + self.b >= 0:
                self.W -= self.X[self.idx,:].reshape(self.W.shape)
                self.b -= 1

            self.idx += 1
            yield loss


if __name__ == '__main__':

    W = np.random.rand(2,1)
    b = np.random.rand(1,1)
    X = np.random.uniform(size=(10,2))
    y = np.concatenate((np.ones((10,1)), np.zeros((10,1))))
    X = np.concatenate((X,-X))

    optimizer = PerceptronLEarning(X,y,W,b)
    losses = list(optimizer.train())

    fig, axs = plt.subplots(figsize=(10,7))
    plt.plot(list(range(len(losses))), losses)
    plt.title("Perceptron learning loss")
    plt.ylabel('Loss')
    plt.xlabel('Epochs')
    plt.savefig('../artifacts/perceptron-learning-loss.png')