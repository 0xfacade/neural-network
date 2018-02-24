import matplotlib.pyplot as plt
import os.path
from neural_network import *
from data_handlers import MnistHandler


def train_network(epochs, batchsize, learning_rate):
    # Construct the layers.
    l1 = Linear(dim_input=28 * 28, num_neurons=10)
    sm = SoftMax()
    ce = CrossEntropy()
    nn = NeuralNetwork([l1, sm], ce)

    X, T = MnistHandler.load_data("train")
    for e in range(epochs):
        nn.train(X, T, batchsize, learning_rate)
        loss, acc = nn.evaluate(X, T)
        print(f"Finshed epoch #{e}. Training loss: {loss}, training acc: {acc}")
    print("Training done. Evaluating on test set.")
    X_test, T_test = MnistHandler.load_data("test")
    loss, acc = nn.evaluate(X_test, T_test)
    print(f"Loss: {loss}, accuracy: {acc}")
    return nn


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

if __name__ == "__main__":
    # Play with these parameters.
    epochs = 100
    batchsize = 600
    learning_rate = 0.1

    # Check if we have already trained a softmax regressor with
    # precisely these parameters, otherwise start training one.
    nn_name = f"trained_networks/softmax-regressor-{epochs}-{batchsize}-{learning_rate}.trained"
    if os.path.exists(nn_name):
        nn = NeuralNetwork.load(nn_name)
    else:
        nn = train_network(epochs, batchsize, learning_rate)
        NeuralNetwork.save(nn, nn_name)

    # Show some statistics on the validation set.
    X_valid, T_valid = MnistHandler.load_data("valid")
    num_samples, _ = X_valid.shape
    loss, acc = nn.evaluate(X_valid, T_valid)
    num_misclassified  = int((1-acc) * num_samples)
    print("Statistics on validation set:")
    print(f"Loss:\t {loss}")
    print(f"Accuracy:\t {acc}")
    print(f"Misclassified: \t {num_misclassified} / {num_samples}")

    # Show actual classification demonstration.
    handcheck_classification(X_valid, T_valid, nn)
