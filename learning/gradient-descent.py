import numpy as np
import matplotlib.pyplot as plt



class GradientDescent:
    def __init__(self, X, y, W, b, lr=1e-3) -> None:
        self.X = X
        self.y = y
        self.W = W
        self.b = b
        self.lr = lr
        self.mask = np.ones((X.shape[0], 1))

    def sigmoid(self):
        return 1/(1+np.exp(-(np.matmul(self.X, self.W)+self.b*self.mask)))

    def linear(self):
        return np.matmul(self.X, self.W) + self.b*self.mask

    def gradient(self):
        fx = self.sigmoid()
        common = (fx-self.y) * fx * (1-fx)
        dw = np.matmul(self.X.transpose(), common)
        db = np.sum(common, axis=0)

        return dw, db

    def loss(self):
        fx = self.sigmoid()

        return np.sum(np.square(self.y-fx), axis=0)

    def train(self, n_epochs):
        for i in range(n_epochs):
            dw, db = self.gradient()
            self.W -= self.lr*dw
            self.b -= self.lr*db

            yield self.loss()


if __name__ == '__main__':

    W = np.random.rand(2,1)
    b = np.random.rand(1,1)
    X = np.random.rand(10,2)
    y = np.random.rand(10,1)
    n_epochs =150

    optimizer = GradientDescent(X,y,W,b, lr=1e-1)
    losses = list(optimizer.train(n_epochs))

    fig, axs = plt.subplots(figsize=(10,7))
    plt.plot(list(range(n_epochs)), losses)
    plt.title("Gradient Descent loss")
    plt.ylabel('Loss')
    plt.xlabel('Epochs')
    plt.savefig('../artifacts/gradient-descent-loss.png')