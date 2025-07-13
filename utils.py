import numpy as np
def load_dataset():
    with np.load("mnist.npz") as f:
        x_train = f['x_train'].astype("float32") / 255
        x_train = x_train.reshape((60000, 784))  # Correct change of form
        y_train = f['y_train']
        y_train = np.eye(10)[y_train]  # One-hot encoding
        return x_train, y_train
