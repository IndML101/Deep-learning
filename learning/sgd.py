import numpy as np
import matplotlib.pyplot as plt


class SGD:
    def __init__(self, X, y, W, b, lr=1e-3) -> None:
        self.X = X
        self.y = y
        self.W = W
        self.b = b
        self.lr = lr

    def sigmoid(self, X):
        return 1/(1+np.exp(-self.linear(X)))

    def linear(self, X):
        return np.dot(X, self.W) + self.b

    def gradient(self, X, y):
        fx = self.sigmoid(X)
        db = (fx-y) * fx * (1-fx)
        dw = X*db

        return dw, db

    def loss(self, X, y):
        fx = self.sigmoid(X)

        return np.square(y-fx)

    def train(self, n_epochs):
        for i in range(n_epochs):
            loss = 0
            for j in range(self.X.shape[0]):
                dw, db = self.gradient(self.X[j], self.y[j])
                self.W -= self.lr*dw.reshape(self.W.shape)
                self.b -= self.lr*db

                loss += self.loss(self.X[j], self.y[j])
            
            yield loss

if __name__ == '__main__':

    W = np.random.rand(2,1)
    b = np.random.rand(1,1)
    X = np.random.rand(10,2)
    y = np.random.rand(10,1)
    n_epochs =150

    optimizer = SGD(X,y,W,b, lr=1e-1)
    losses = np.fromiter(optimizer.train(n_epochs), dtype=np.float64)
    losses = losses.flatten()

    fig, axs = plt.subplots(figsize=(10,7))
    plt.plot(list(range(len(losses))), losses)
    plt.title("SGD loss")
    plt.ylabel('Loss')
    plt.xlabel('Steps')
    plt.savefig('../artifacts/sgd-loss.png')