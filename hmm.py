from typing import Tuple

import nltk
import numpy as np


class HiddenMarkovModel:

    DECODING = ["global", "local"]

    def __init__(self, A: np.array = None, B: np.array = None, state_map: dict = None) -> None:
        """_summary_

        Args:
            A (np.array, optional): Transition probability matrix. Defaults to None.
            B (np.array, optional): Emission probability matrix. Defaults to None.
        """
        self.A = A
        self.B = B
        self._state_map = state_map

    # fit
    # TODO make invariant to the number of passed features!!!!
    def estimate_parameters(self,
                            X: list[list[dict]],
                            Y: list[list[str]],
                            smoothing: nltk.ProbDistI = nltk.WittenBellProbDist,
                            kwargs: dict = {"bins": 1e5}) -> None:
        self.A = self._get_transitions(Y, smoothing, kwargs)
        self.B = self._get_emissions(X, smoothing, kwargs)

    def decode(self, decoding):
        # Find most likely state (local) or sequence of states (global) given a sequence of observations
        if decoding.lower() not in self.DECODING:
            raise NotImplementedError(
                "Unknown decoding type. Choose from ['global', 'local']")

    def viterbi(self, observations: list[str]) -> Tuple(list[str], float):
        # method to find a global decoding
        # TODO sort out shapes of A and B
        N = len(self.A)
        T = len(observations)
        V, B = np.zeros(shape=[N, T]), np.zeros(shape=[N, T], dtype=int)

        # initialize
        # state == 0 is <S> and state == N is </S>
        for s in range(1, N-1):  # 1-17
            V[s, 0] = self.A[0, s] * self.B[s].prob(observations[0])
            B[s, 0] = 0
        for t in range(1, T):
            for s in range(1, N-1):
                # probabilities for each state given the path
                probs = np.array([0] + [V[s_, t-1] * self.A[s_, s]
                                 for s_ in range(1, N-1)] + [0])
                V[s, t] = self.B[s].prob(observations[t]) * np.max(probs)
                B[s, t] = np.argmax(probs)

        best_path_probability = np.max(V[:, T-1])
        best_path_pointer = np.argmax(V[:, T-1])
        best_path = [best_path_pointer]
        for i in range(T-1, 0, -1):
            best_path.insert(0, B[best_path[0], i])

        predictions = [self._state_map[i] for i in best_path]

        return predictions, best_path_probability

    def _get_transitions(self, Y: list[list[str]], smoothing: nltk.ProbDistI, kwargs: dict) -> np.array:
        pos_tags = list(set([word for sen in Y for word in sen]))

        # Get occureces counts across the sample
        pos_frequencies = {i: nltk.FreqDist()
                           for i in pos_tags + ["<S>", "</S>"]}

        for sentence in Y:
            # get start of sentence transitions
            pos_frequencies["<S>"][sentence[0]] += 1
            # get end of sentence transitions
            pos_frequencies[sentence[-1]]["</S>"] += 1
            # get other transitions
            for i in range(1, len(sentence)-1):
                pos_frequencies[sentence[i]][sentence[i+1]] += 1

        # Normalize to get probabilities
        A = {i: smoothing(pos_frequencies[i], **kwargs)
             for i in pos_tags + ["<S>", "</S>"]}

        # return A
        return self._dist_dict_to_array(A)

    def _get_emissions(self, Y: list[list[dict]], smoothing: nltk.ProbDistI, kwargs: dict) -> list[nltk.ProbDistI]:
        emissions = [(word['upos'], word['lemma'])
                     for sen in Y for word in sen]

        B_dict = {tag: smoothing(nltk.FreqDist([w for (t, w) in emissions if t == tag]),
                                 **kwargs)
                  for tag in set([t for (t, _) in emissions])}

        B = [0] * len(self._state_map)
        for i, v in self._state_map.items():
            if v not in ["<S>", "</S>"]:
                B[i] = B_dict[v]

        return B

    def _dist_dict_to_array(self, dist_dict: dict) -> np.array:
        N = len(dist_dict.keys())
        arr = np.zeros([N, N])
        if not self._state_map:
            # self._state_map = {k: i+1 for i, k in enumerate(
            #     [k for k in dist_dict.keys() if k not in ["<S>", "</S>"]])}
            # self._state_map["<S>"] = 0
            # self._state_map["</S>"] = N-1
            self._state_map = {i+1: k for i, k in enumerate(
                [k for k in dist_dict.keys() if k not in ["<S>", "</S>"]])}
            self._state_map[0] = "<S>"
            self._state_map[N-1] = "</S>"

        for i, v in self._state_map.items():
            for j, v_ in self._state_map.items():
                arr[i][j] = dist_dict[v].prob(v_)
        return arr


def prune_sentence(sent):
    return [token for token in sent if type(token['id']) is int]


def conllu_corpus(path):
    data_file = open(path, 'r', encoding='utf-8')
    sents = list(parse_incr(data_file))
    return [prune_sentence(sent) for sent in sents]


if __name__ == "__main__":
    from conllu import parse_incr
    train_corpus_path = 'treebanks/UD_English-GUM/en_gum-ud-train.conllu'
    train_sents = conllu_corpus(train_corpus_path)

    m = HiddenMarkovModel()
    d = m.estimate_parameters(
        train_sents, [[i['upos'] for i in s] for s in train_sents])
    m.viterbi(["Don't", "lose", "focus"])

    # A_counts = self._count_occurences(Y)
    # pos_tags = [i for i in A_counts.keys() if i not in ["<S>", "<\S>"]]

    # # normalize to get probabilities
    # A = { i: { j :  A_counts[i][j]/A_counts[i]["TOTAL"] \
    #                 for j in pos_tags} \
    #                 for i in A_counts.keys()}
    # return A

    # # check if we have empty categories
    # removed = []
    # for k in A_counts.keys():
    #     if A_counts[k]["TOTAL"] == 0:
    #         # remove rows
    #         A_counts.pop(k)
    #         removed.append(k)
    # # remove columnd
    # if removed:
    #     for k in A_counts.keys():
    #         for r in removed:
    #             A_counts[k].pop(r)

    # def _count_occurences(self, Y: list[list[str]]) -> dict:
    #     pos_tags = list(set([word for sen in Y for word in sen]))

    #     A_counts = { i: { j : 0  for j in pos_tags + ["TOTAL"] } \
    #                              for i in pos_tags + ["<S>", "<\S>"]}

    #     A_counts["<S>"]["TOTAL"], A_counts["<\S>"]["TOTAL"]  = len(Y), len(Y)

    #     for sentence in Y:
    #         # get start of sentence transitions
    #         A_counts["<S>"][sentence[0]] += 1
    #         # get end of sentence transitions
    #         A_counts["<\S>"][sentence[-1]] += 1
    #         # get other transitions
    #         for i in range(1,len(sentence)-1):
    #             A_counts[sentence[i]][sentence[i+1]] += 1
    #             A_counts[sentence[i]]["TOTAL"] += 1

    #     return A_counts
