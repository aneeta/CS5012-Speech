import matplotlib.pyplot as plt
import sklearn.metrics as sm
import numpy as np


def evaluate(predictions, labels):
    eval = {
        "Accuracy": np.mean([sm.accuracy_score(labels[i], predictions[i]) for i in range(len(labels))]),
        "Precision": np.mean([sm.precision_score(labels[i], predictions[i], average='micro') for i in range(len(labels))]),
        # "Recall": np.mean([sm.recall_score(labels[i], predictions[i]) for i in range(len(labels))]),
        # "F1": np.mean([sm.f1_score(labels[i], predictions[i]) for i in range(len(labels))]),
    }
    return eval
