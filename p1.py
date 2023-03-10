import warnings

import preprocessing
import hmm
import evaluation
import utils


def main():
    # parse command line arguments
    args = utils.get_cli_args()

    # warnings
    if not args['warnings']:
        warnings.filterwarnings("ignore")  # TODO less general filter
        # np.seterr(divide='ignore')
        # warnings.filterwarnings('ignore', module='nltk')

    # results object
    res = {}
    acc = {}
    for l in args['langs']:
        lang = l.upper()
        print("Tagged language:", lang)

        # preprocess data
        train, test = preprocessing.get_corpus(lang)
        X, Y = preprocessing.format_corpus(train)
        _, Y_t = preprocessing.format_corpus(test, test=True)
        test_sentences = [[w for (t, w) in sen] for sen in Y_t]
        test_labels = [[t for (t, w) in sen] for sen in Y_t]

        if args['unk']:
            X, mapping = preprocessing.replace_unk(X, lang)
            test_sentences = [[mapping[w] if w in mapping.keys(
            ) else w for w in sen] for sen in test_sentences]

        m = hmm.HiddenMarkovModel()

        # train
        m.estimate_parameters(X, Y, hmm.SMOOTHING[args['smoothing']])

        # predict
        predictions, _ = m.predict_viterbi(test_sentences)

        # evaluate
        res[lang], acc[lang] = evaluation.evaluate(
            lang, predictions, test_labels, args['plot'])
        print("Accuracy:", acc[lang])
    res_df = utils.format_results(res)
    if args['csv']:
        res_df.to_csv(args['csv'], index=False)
    print("Accuracies:")
    print(",".join(acc.keys()))
    print(",".join([str(i) for i in acc.values()]))


if __name__ == "__main__":
    main()
