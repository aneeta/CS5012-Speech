import argparse

import preprocessing
import hmm
import evaluation


def get_cli_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-l', '--langs',
        nargs='+',
        default=preprocessing.LANGUAGES,
        help="""Choose a language from ${}.""".format(preprocessing.LANGUAGES))
    parser.add_argument(
        '-u', '--unk',
        action='store_true')
    parser.add_argument(
        '-s', '--smoothing',
        default="WB",
        help="""Choose smoothing method from ${}.""".format(hmm.SMOOTHING))
    args = parser.parse_args()
    return vars(args)


def main():
    args = get_cli_args()

    res = {}

    for lang in args['langs']:
        # preprocess data
        train, test = preprocessing.get_corpus(lang)
        X, Y = preprocessing.format_corpus(train)
        _, Y_t = preprocessing.format_corpus(test, test=True)

        if args['unk']:
            Y = preprocessing.replace_unk(Y, lang)
            Y_t = preprocessing.replace_unk(Y_t, lang)

        test_sentences = [[w for (t, w) in sen] for sen in Y_t]
        test_labels = [[t for (t, w) in sen] for sen in Y_t]

        m = hmm.HiddenMarkovModel()
        # train
        m.estimate_parameters(X, Y, hmm.SMOOTHING[args['smoothing']])
        # test
        predictions, best_path_probabilities = m.viterbi(test_sentences[:10])
        res[lang] = evaluation.evaluate(predictions[:10], test_labels[:10])
        print(lang, res[lang])


if __name__ == "__main__":
    main()
