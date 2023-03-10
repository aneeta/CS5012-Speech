import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np
import sklearn.metrics as sm
from sklearn.preprocessing import MultiLabelBinarizer

from utils import get_cli_vis


def evaluate(lang, predictions, labels, plot):
    tags = list(set([t for sen in labels for t in sen]))
    N = len(tags)
    confusion_matrix = np.zeros([N, N], dtype=int)
    reports = []

    for i in range(len(labels)):
        # mlb = MultiLabelBinarizer()
        # mlb.fit(predictions[i])
        # encoded_pred = mlb.transform(predictions[i])
        # encoded_labels = mlb.transform(labels[i])
        # categories = mlb.classes_
        confusion_matrix += sm.confusion_matrix(
            labels[i], predictions[i], labels=tags)

        reports.append(sm.classification_report(predictions[i], labels[i],
                                                # target_names=tags,
                                                output_dict=True))
    eval_dict, accuracy = unify_reports(reports)
    if plot:
        plot_confusion_matrix(lang, plot, confusion_matrix, tags)
        # plot_confusion_matrices(
        #     lang, plot, encoded_labels, encoded_pred, categories)
    return eval_dict, accuracy


def unify_reports(reports: list[dict]):
    res = {}
    samples = {}
    # accuracy logged as numeric rather than dict, has to be handled separately
    accuracy = sum([i['accuracy'] for i in reports])/len(reports)
    for r in reports:
        # sum up all values
        for k, v in r.items():
            if k in res:
                for k_, v_ in v.items():
                    res[k][k_] += v_
                samples[k] += 1
            elif k != 'accuracy':
                res[k] = v
                samples[k] = 1
    # divide by number of samples
    for k, v in res.items():
        for k_, v_ in v.items():
            if k_ != 'support':
                res[k][k_] = v_/samples[k]
    return res, accuracy


# def plot_multi_confusion_matrices(lang, fig_prefix, labels, predictions, names, n_cols=5):
#     c_m = sm.multilabel_confusion_matrix(labels, predictions)
#     n_rows = round(len(names) / n_cols)
#     fig, ax = plt.subplots(n_rows, n_cols)
#     ax = ax.flatten()
#     # remove unnecerrasy subplots
#     for i in range(len(names), len(ax)):
#         fig.delaxes(ax[i])
#     # plot
#     for i, a in enumerate(ax):
#         sub_m = c_m[i]
#         a.imshow(sub_m, cmap=plt.cm.Blues, alpha=0.5)
#         for j in range(sub_m.shape[0]):
#             for k in range(sub_m.shape[1]):
#                 # text
#                 a.text(x=k, y=j, s=sub_m[k, j],
#                        va='center', ha='center', size='smaller')
#         a.set_title(names[i])
#         a.set_xticklabels(["_", "P", "N"])
#         a.set_yticklabels(["_", "T", "F"])
#     fig.suptitle("Confusion matrix for %s" % lang)
#     fig.tight_layout()

#     plt.savefig("%s_confusion_matrix_%s.png" % (fig_prefix, lang))


def plot_confusion_matrix(lang, fig_prefix, matrix, labels):
    fig, ax = plt.subplots(figsize=(8, 8))
    p = ax.imshow(matrix, cmap=plt.cm.plasma,
                  #   norm=colors.CenteredNorm(),
                  alpha=0.33)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            # text
            ax.text(x=i, y=j, s=matrix[j, i],
                    va='center', ha='center', size='smaller')
    ax.set_title("Confusion matrix for %s" % lang)
    ax.set_xticks(np.arange(len(matrix)))
    ax.set_xticklabels(labels, rotation=90)
    ax.set_yticks(np.arange(len(matrix)))
    ax.set_yticklabels(labels)
    fig.colorbar(p, fraction=0.046, pad=0.04)
    # plt.setp(ax.get_xticklabels(), visible=True)  # , rotation=30, ha='right')
    fig.tight_layout()
    plt.savefig("%s_confusion_matrix_%s.png" % (fig_prefix, lang))


def visualise_results(csv_file):
    # TODO
    pass


if __name__ == "__main__":
    args = get_cli_vis()
    visualise_results(args['csv'])
