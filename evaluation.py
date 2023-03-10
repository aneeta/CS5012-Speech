import matplotlib.pyplot as plt
import numpy as np
import sklearn.metrics as sm


def evaluate(lang, predictions, labels, plot):
    tags = list(set([t for sen in labels for t in sen]))
    N = len(tags)
    confusion_matrix = np.zeros([N, N], dtype=int)
    reports = []

    # for each sample (sentence)
    for i in range(len(labels)):
        confusion_matrix += sm.confusion_matrix(
            labels[i], predictions[i], labels=tags)

        reports.append(sm.classification_report(predictions[i], labels[i],
                                                # target_names=tags,
                                                output_dict=True))
    eval_dict, accuracy = unify_reports(reports)

    if plot:
        plot_confusion_matrix(lang, plot, confusion_matrix, tags)
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


def plot_confusion_matrix(lang, fig_prefix, matrix, labels):
    fig, ax = plt.subplots(figsize=(8, 8))
    p = ax.imshow(matrix, cmap=plt.cm.plasma,
                  alpha=0.33)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            # number annotations
            ax.text(x=i, y=j, s=matrix[j, i],
                    va='center', ha='center', size='smaller')
    ax.set_title("Confusion matrix for %s" % lang)
    ax.set_xticks(np.arange(len(matrix)))
    ax.set_xticklabels(labels, rotation=90)
    ax.set_yticks(np.arange(len(matrix)))
    ax.set_yticklabels(labels)
    fig.colorbar(p, fraction=0.046, pad=0.04)
    fig.tight_layout()
    plt.savefig("%s_confusion_matrix_%s.png" % (fig_prefix, lang))
