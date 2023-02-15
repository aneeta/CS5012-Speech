import pandas as pd
from typing import Tuple

from conllu import parse_incr
from conllu.models import Token

import re

from nltk import FreqDist

LANGUAGES = ["EN", "FR", "UK"]

FILEPATHS = {
    "EN": "treebanks/UD_English-GUM/en_gum",
    "FR": "treebanks/UD_French-Rhapsodie/fr_rhapsodie",
    "UK": "treebanks/UD_Ukrainian-IU/uk_iu"
}


def get_corpus(lang: str) -> list:
    corpus = []  # train and test data
    for i in ['-ud-train.conllu', '-ud-test.conllu']:
        data_file = open(FILEPATHS[lang]+i, 'r', encoding='utf-8')
        corpus.append([prune_sentence(sent)
                      for sent in list(parse_incr(data_file))])
    return corpus


def prune_sentence(sent) -> list:
    return [token for token in sent if type(token['id']) is int]


# def format_corpus(corpus: list[list[Token]]) -> list[Tuple[str]]:
#     return [(word['upos'], word['lemma']) for sen in corpus for word in sen]

# -> Tuple(list[Tuple[str]], list[list[str]]):
def format_corpus(corpus: list[list[Token]], test=False):
    # emissions
    X = [(word['upos'], word['lemma']) for sen in corpus for word in sen]
    # transitions
    Y = [[(word['upos'], word['lemma']) for word in sen] for sen in corpus] if test else \
        [[word['upos'] for word in sen] for sen in corpus]
    return X, Y


def replace_unk(corpus: list[Tuple[str]], lang: str):
    if lang.upper() not in LANGUAGES:
        raise NotImplementedError(
            """Unknown language selected.
               Choose from ["EN", "FR", "UK"]""")
    if lang.upper() == "EN":
        return unk_tagging_en(corpus)
    if lang.upper() == "FR":
        return unk_tagging_fr(corpus)
    if lang.upper() == "UK":
        return unk_tagging_uk(corpus)


def unk_tagging_en(corpus: list[Tuple[str]]) -> list[Tuple[str]]:
    COMMON_ENDS = {
        2: ["ed", "er", "ly", "ty", "ry", "al", "el", "an", "en", "or", "ic", "se"],
        3: ["ing", "ist", "ate", "ous", "ent", "ect", "eur", "ess", "ery"],
        4: ["able", "ment", "tion", "tive", "ship", "ness"]
    },

    unk_copus = [(t, _replace_email(w)) for (t, w) in corpus]
    hapaxes_map = {i: "UNK" for i in FreqDist(
        w for (t, w) in unk_copus).hapaxes()}
    for k in hapaxes_map.keys():
        hapaxes_map[k] = _replace_proper_name(k)
        hapaxes_map[k] = _replace_acronym(k)
        hapaxes_map[k] = _replace_numeric(k)
        for i in [4, 3, 2]:
            if k[-i:] in COMMON_ENDS[i]:
                hapaxes_map[k] = _replace_stem(k, i)
                break
    return [(t, _map_word(w, hapaxes_map)) for (t, w) in unk_copus]

    # unk_copus = [(t, _replace_proper_name(w)) for (t,w) in corpus]
    # unk_copus = [(t, _replace_stem(w, len(i))) for (t,w) in unk_copus for i in \
    #              ["able", "ment", "tion", "tive", "ship", "ness",
    #               "ing", "ist", "ate", "ous", "ent", "ect", "eur", "ess", "ery",
    #               "ed", "er", "ly", "ty", "ry", "al", "el", "an", "en", "or", "ic", "se"] \
    #  if w[-len(i):] == i]


def unk_tagging_fr(corpus: list[Tuple[str]]) -> list[Tuple[str]]:
    pass


def unk_tagging_uk(corpus: list[Tuple[str]]) -> list[Tuple[str]]:
    pass


def _replace_email(email: str) -> str:
    return re.sub("[\w]+@([\w]+\.[A-Za-z]+)+", "UNK@UNK", email)


def _replace_acronym(name: str) -> str:
    return re.sub("[A-Z]{2,6}", "UNK@UNK", name)


def _replace_proper_name(name: str) -> str:
    return re.sub("(([A-Z][a-z]+)-?)?[A-Z][a-z]", "PROPERNAME", name)


def _replace_numeric(name: str, end: int) -> str:
    return re.sub("\d+(st|nd|rd|th)?", "NUMERIC", name)


def _replace_stem(name: str, end: int) -> str:
    return "UNK" + name[-end:]


def _replace_stem(name: str, end: int) -> str:
    return "UNK" + name[-end:]


def _map_word(word: str, map: dict) -> str:
    if word in map.keys():
        return map[word]
    return word
