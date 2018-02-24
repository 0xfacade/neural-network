import numpy as np
import os.path
import pickle
import matplotlib.pyplot as plt


class Linear:

    def __init__(self, dim_input, num_neurons):
        self.dim_input, self.num_neurons = dim_input, num_neurons
        self.W = np.random.normal(0, np.sqrt(2 / (dim_input + num_neurons)), (dim_input, num_neurons))
        self.b = np.zeros((1, num_neurons))

    def fprop(self, X):
        self.X = X
        self.Z = X @ self.W + self.b
        return self.Z

    def bprop(self, V):
        num_samples, input_dim = self.X.shape
        _, output_dim = self.W.shape
        self.db = np.sum(V, axis=0) / num_samples
        self.dW = np.zeros((input_dim, output_dim))
        # TODO: do this more efficiently
        for r in range(num_samples):
            x = self.X[r, :]
            v = V[r,:]
            dWx = np.tile(x, (output_dim, 1)).transpose()
            self.dW += v * dWx
        self.dW /= num_samples
        return V @ self.W.transpose()

    def update_params(self, learning_rate):
        self.W -= learning_rate * self.dW
        self.b -= learning_rate * self.db

class SoftMax:

    def fprop(self, X):
        Exp = np.exp(X)
        row_sums = np.sum(Exp, axis=1)
        self.Sig = (Exp.transpose() / row_sums).transpose()
        return self.Sig

    def bprop(self, V):
        num_samples, num_inputs = V.shape
        R = np.zeros((num_samples, num_inputs))
        for r in range(num_samples):
            v = V[r, :]
            sig = self.Sig[r, :]
            R[r, :] = sig * (v - np.inner(v, sig))
        return R

    def update_params(self, learning_rate):
        pass

    @staticmethod
    def test():
        sm = SoftMax()
        X = np.array([[1, 3, 4], [0.1, 0.2, 0.3], [0.5, 0.7, 10], [0.3, 0.5, 2]])
        Z = sm.fprop(X)
        num_samples, _ = X.shape
        r = np.sum(Z, axis=1) - np.ones(num_samples)
        if np.sum(r) > 0.001:
            print("Rows don't sum to one!")
        R = sm.bprop(Z)
        print("Data: ")
        print(X)
        print("Softmax:")
        print(Z)
        print("Bprop:")
        print(R)


class CrossEntropy:

    def set_target(self, T):
        self.T = T

    def fprop(self, Z):
        # The rows z of Z are the outputs of the softmax layer
        # (different rows belong to different data points)
        # If input x generates activation z and belongs to class t,
        # then really the cross entropy is just -ln(z_t).
        self.num_samples, self.input_dim = Z.shape
        self.z_t = Z[np.arange(self.num_samples), self.T]
        ce = - np.log(self.z_t)
        return ce

    def bprop(self):
        R = np.zeros((self.num_samples, self.input_dim))
        R[np.arange(self.num_samples), self.T] = -1 / self.z_t
        return R

    def update_params(self, learning_rate):
        pass

    @staticmethod
    def test():
        Z = np.array([[0.1, 0.2, 0.7], [0.2, 0.3, 0.5], [0.6, 0.3, 0.1], [0.9, 0.05, 0.05]])
        T = np.array([0, 1, 2, 1])
        ce = CrossEntropy()
        ce.set_target(T)
        l = ce.fprop(Z)
        R = ce.bprop()
        print("Activation:")
        print(Z)
        print("Target:")
        print(T)
        print("Cross entropy loss: ")
        print(l)
        print("Derivative with respect to input: ")
        print(R)


class NeuralNetwork:

    def __init__(self, layers, loss):
        self.layers = layers
        self.loss = loss

    @staticmethod
    def _batchify(X, T, batchsize):
        start, stop = 0, batchsize
        num_samples, _ = X.shape
        while start < num_samples:
            if stop >= num_samples:
                stop = num_samples
            yield X[start:stop, :], T[start:stop]
            start, stop = stop, stop + batchsize

    def _forward(self, X):
        z = X
        for layer in self.layers:
            z = layer.fprop(z)
        return z

    def _backward(self):
        dz = self.loss.bprop()
        for layer in reversed(self.layers):
            dz = layer.bprop(dz)

    def _update(self, learning_rate):
        # update parameters
        for layer in self.layers:
            layer.update_params(learning_rate)

    def train(self, X, T, batchsize, learning_rate):
        # process the training data in batches
        for Xb, Tb in self._batchify(X, T, batchsize):
            z = self._forward(Xb)
            self.loss.set_target(Tb)
            # TODO: maybe do something with the loss?
            self.loss.fprop(z)
            self._backward()
            self._update(learning_rate)

    def classify(self, X):
        Z = self._forward(X)
        hard_labels = np.argmax(Z, axis=1)
        return hard_labels

    def evaluate(self, X, T):
        Z = self._forward(X)
        self.loss.set_target(T)
        l = np.average(self.loss.fprop(Z))

        num_samples, _ = X.shape
        hard_labels = self.classify(X)
        correct_labels = np.sum(hard_labels == T)
        acc = correct_labels / num_samples

        return l, acc

    @staticmethod
    def save(nn, filename):
        with open(filename, "wb") as out:
            pickle.dump(nn, out)

    @staticmethod
    def load(filename):
        with open(filename, "rb") as i:
            return pickle.load(i)


def load_data(identifier):
    data_file = f"mnist-{identifier}-data"
    labels_file = f"mnist-{identifier}-labels"
    if not os.path.exists(f"{data_file}.npy") \
            or not os.path.exists(f"{labels_file}.npy"):
        X = np.genfromtxt(f"{data_file}.csv", delimiter=" ", dtype=np.float32) / 255
        T = np.genfromtxt(f"{labels_file}.csv", delimiter=" ", dtype=np.int8)
        np.save(data_file, X)
        np.save(labels_file, T)
    else:
        X = np.load(f"{data_file}.npy")
        T = np.load(f"{labels_file}.npy")
    return X, T


def handcheck_classification(X, T, nn):
    num_samples, _ = X.shape
    R = nn.classify(X)

    for i in range(num_samples):
        x = X[i, :]
        t = T[i]
        c = R[i]
        print(f"Correct: {t}, classified as: {c}")
        plt.imshow(x.reshape((28, 28)))
        plt.show()


def train_linear_softmax_ce(epochs, batchsize, learning_rate):
    nn_name = f"l-sm-ce-{epochs}-{batchsize}-{learning_rate}.trained"

    if os.path.exists(nn_name):
        nn = NeuralNetwork.load(nn_name)
    else:
        l1 = Linear(dim_input=28 * 28, num_neurons=10)
        sm = SoftMax()
        ce = CrossEntropy()
        nn = NeuralNetwork([l1, sm], ce)

        X, T = load_data("train")
        for e in range(epochs):
            nn.train(X, T, batchsize, learning_rate)
            loss, acc = nn.evaluate(X, T)
            print(f"Finshed epoch #{e}. Training loss: {loss}, training acc: {acc}")
        print("Training done. Evaluating on test set.")
        X_test, T_test = load_data("test")
        loss, acc = nn.evaluate(X_test, T_test)
        print(f"Loss: {loss}, accuracy: {acc}")

        NeuralNetwork.save(nn, nn_name)
    return nn

if __name__ == "__main__":
    epochs = 100
    batchsize = 600
    learning_rate = 0.1

    nn = train_linear_softmax_ce(epochs, batchsize, learning_rate)
    X_valid, T_valid = load_data("valid")
    num_samples, _ = X_valid.shape
    loss, acc = nn.evaluate(X_valid, T_valid)
    misclassified = int((1 - acc) * num_samples)
    print(f"{misclassified} samples were misclassified.")









