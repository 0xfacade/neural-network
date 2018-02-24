import os.path
import numpy as np


class MnistHandler:

    @staticmethod
    def load_data(identifier, data_dir="mnist"):
        """Loads the MNIST data set from disk. Returns:
                * design matrix X
                * vector of correct labels T
            Assumes the data files follow the pattern mnist-identifier-data,
            where identifier is one of train, valid, test.
            The files can be either in CSV format or serialized numpy arrays.
        """

        data_file = os.path.join(data_dir, f"mnist-{identifier}-data")
        labels_file = os.path.join(data_dir, f"mnist-{identifier}-labels")

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
