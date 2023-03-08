from typing import Tuple

import nltk
import numpy as np

SMOOTHING = {
    "WB": nltk.WittenBellProbDist,
    "GT": nltk.SimpleGoodTuringProbDist
    # "KN": nltk.KneserNeyProbDist # needs trigram
}


class HiddenMarkovModel:

    def __init__(self, A: np.array = None, B: np.array = None, pi: np.array = None, state_map: dict = None) -> None:
        """_summary_

        Args:
            A (np.array, optional): Transition probability matrix. Defaults to None.
            B (np.array, optional): Emission probability matrix. Defaults to None.
        """
        self.A = A  # transitions
        self.B = B  # emissions
        self.pi = pi  # initial states
        self._state_map = state_map

    # fit
    # TODO extend to n-grams??
    def estimate_parameters(self,
                            X: list[Tuple[str]],  # emissions
                            Y: list[list[str]],  # transitions
                            smoothing: nltk.ProbDistI = nltk.WittenBellProbDist,
                            kwargs: dict = {"bins": 1e5}) -> None:
        self.A, self.pi = self._get_transitions(Y, smoothing, kwargs)
        self.B = self._get_emissions(X, smoothing, kwargs)

    def predict_viterbi(self, samples: list[list[str]]):
        predictions = []
        best_path_probabilities = []
        for s in samples:
            pred, path = self._viterbi(s)
            predictions.append(pred)
            best_path_probabilities.append(path)
        return predictions, best_path_probabilities

    def _viterbi(self, observations: list[str]):  # -> Tuple(list[str], float):
        # method to find a global decoding

        N, T = len(self.A), len(observations)
        # Viterbi and backpointers matrices
        V, B = np.zeros(shape=[N, T]), np.zeros(shape=[N, T], dtype=int)

        # initialize (T=0)
        for s in range(N):
            V[s, 0] = np.exp(np.log(self.pi[s]) +
                             np.log(self.B[s].prob(observations[0])))
            B[s, 0] = 0
        # populate
        for t in range(1, T):
            for s in range(N):
                # probabilities for each state given the path
                probs = np.exp(np.log(V[:, t-1]) + np.log(self.A[:, s]))
                V[s, t] = np.exp(np.log(self.B[s].prob(
                    observations[t])) + np.log(np.max(probs)))
                B[s, t] = np.argmax(probs)
        # terminate
        best_path_probability = np.max(V[:, T-1])
        best_path_pointer = np.argmax(V[:, T-1])
        best_path = [best_path_pointer]
        for i in range(T-1, 0, -1):
            best_path.insert(0, B[best_path[0], i])

        prediction = [self._state_map[i] for i in best_path]
        return prediction, best_path_probability

    def _get_transitions(self, Y: list[list[str]], smoothing: nltk.ProbDistI, kwargs: dict) -> np.array:
        if not self._state_map:
            self._state_map = {i: v for i, v in enumerate(
                list(set([i for sen in Y for i in sen])))}

        pos_tags = list(self._state_map.values())

        # Get occureces counts across the sample
        pos_frequencies = {i: nltk.FreqDist()
                           for i in pos_tags}
        starting_frequencies = nltk.FreqDist()
        for sentence in Y:
            # get start of sentence transitions
            starting_frequencies[sentence[0]] += 1
            # get other transitions
            for i in range(len(sentence)-1):
                pos_frequencies[sentence[i]][sentence[i+1]] += 1

        # Normalize to get probabilities
        A = {i: smoothing(pos_frequencies[i], **kwargs)
             for i in pos_tags}
        pi = smoothing(starting_frequencies, **kwargs)
        pi_array = np.array([pi.prob(i) for i in pos_tags])

        return self._dist_to_array(A), pi_array

    def _get_emissions(self, X: list[Tuple[str]], smoothing: nltk.ProbDistI, kwargs: dict) -> list[nltk.ProbDistI]:

        B_dict = {tag: smoothing(nltk.FreqDist([w for (t, w) in X if t == tag]),
                                 **kwargs)
                  for tag in set([t for (t, _) in X])}

        B = [0] * len(self._state_map)
        for i, v in self._state_map.items():
            B[i] = B_dict[v]

        return B

    def _dist_to_array(self, dist_dict: dict) -> np.array:
        N = len(dist_dict.keys())
        arr = np.zeros([N, N])

        for i, v in self._state_map.items():
            for j, v_ in self._state_map.items():
                arr[i][j] = dist_dict[v].prob(v_)
        return arr
