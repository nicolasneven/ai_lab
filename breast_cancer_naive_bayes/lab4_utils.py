import numpy as np

data_filename = "data.npy"

feature_names = [
    "age_group",
    "menopause",
    "tumor_size",
    "inv_nodes",
    "node_caps",
    "deg_malig",
    "side",
    "quadrant",
    "irradiated",
]


def accuracy_score(true_labels, predicted_labels):
    return np.average((np.array(true_labels) == np.array(predicted_labels)).astype(int))


def load_data():
    training_inputs, testing_inputs, training_labels, testing_labels = np.load(
        data_filename, allow_pickle=True
    )
    return training_inputs, testing_inputs, training_labels, testing_labels
