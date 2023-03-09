from datetime import datetime

import matplotlib.pyplot as plt
import matplotlib.colors as colors

import sklearn.metrics as sm
from sklearn.preprocessing import MultiLabelBinarizer
import numpy as np


def evaluate(lang, predictions, labels, plot):
    mlb = MultiLabelBinarizer()
    mlb.fit(predictions)
    encoded_pred = mlb.transform(predictions)
    encoded_labels = mlb.transform(labels)
    categories = mlb.classes_

    eval = sm.classification_report(encoded_pred, encoded_labels,
                                    target_names=categories,
                                    output_dict=True)
    if plot:
        plot_confusion_matrices(
            lang, plot, encoded_labels, encoded_pred, categories)
    # print(sm.multilabel_confusion_matrix(mlb.transform(labels), mlb.transform(predictions), labels=mlb.classes_))
    # eval = {
    #     "Accuracy": np.mean([sm.accuracy_score(labels[i], predictions[i]) for i in range(len(labels))]),
    #     "Precision": np.mean([sm.precision_score(labels[i], predictions[i], average='micro') for i in range(len(labels))]),
    #     "Recall": np.mean([sm.recall_score(labels[i], predictions[i]) for i in range(len(labels))]),
    #     "F1": np.mean([sm.f1_score(labels[i], predictions[i]) for i in range(len(labels))]),
    # }
    return eval


def visualise():
    # TODO
    pass


def plot_confusion_matrices(lang, fig_prefix, labels, predictions, names, n_cols=5):
    c_m = sm.multilabel_confusion_matrix(labels, predictions)
    n_rows = round(len(names) / n_cols)
    fig, ax = plt.subplots(n_rows, n_cols)
    ax = ax.flatten()
    # remove unnecerrasy subplots
    for i in range(len(names), len(ax)):
        fig.delaxes(ax[i])
    # plot
    for i, a in enumerate(ax):
        sub_m = c_m[i]
        a.imshow(sub_m, cmap=plt.cm.Blues, alpha=0.5)
        for j in range(sub_m.shape[0]):
            for k in range(sub_m.shape[1]):
                # text
                a.text(x=k, y=j, s=sub_m[k, j],
                       va='center', ha='center', size='smaller')
        a.set_title(names[i])
        a.set_xticklabels(["_", "P", "N"])
        a.set_yticklabels(["_", "T", "F"])
    fig.suptitle("Confusion matrix for %s" % lang)
    fig.tight_layout()

    # timestamp = datetime.now().strftime("%d%m%Y-%H%M%S")

    plt.savefig("%s_confusion_matrix_%s.png" % (fig_prefix, lang))
