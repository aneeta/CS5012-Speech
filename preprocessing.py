import re
from typing import Tuple

from conllu import parse_incr
from conllu.models import Token
from nltk import FreqDist

LANGUAGES = ["EN", "FR", "UK", "PL", "KO"]
SUFFIX_LEN = [4, 3, 2]

FILEPATHS = {
    "EN": "treebanks/UD_English-GUM/en_gum",
    "FR": "treebanks/UD_French-Rhapsodie/fr_rhapsodie",
    "UK": "treebanks/UD_Ukrainian-IU/uk_iu",
    "PL": "treebanks/UD_Polish-LFG/pl_lfg",
    "KO": "treebanks/UD_Korean-Kaist/ko_kaist"
}


def get_corpus(lang: str) -> list:
    """
    Utility function to retrieve a chosen language corpus
    from a conllu file. 

    Args:
        lang (str): corpus language

    Returns:
        list: corpus as a list of sentences (list) of words (Tokens)
    """
    corpus = []  # train and test data
    for i in ['-ud-train.conllu', '-ud-test.conllu']:
        data_file = open(FILEPATHS[lang]+i, 'r', encoding='utf-8')
        corpus.append([prune_sentence(sent)
                      for sent in list(parse_incr(data_file))])
    return corpus


def prune_sentence(sent) -> list:
    # Remove contractions such as "isn't".
    return [token for token in sent if type(token['id']) is int]


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
               Choose from ["EN", "FR", "UK", "PL", "KO"]""")
    if lang.upper() == "EN":
        return unk_tagging_en(corpus)
    if lang.upper() == "FR":
        return unk_tagging_fr(corpus)
    if lang.upper() == "UK":
        return unk_tagging_uk(corpus)
    if lang.upper() == "PL":
        return unk_tagging_pl(corpus)
    if lang.upper() == "KO":
        return unk_tagging_ko(corpus)


def unk_tagging_en(corpus: list[Tuple[str]]) -> list[Tuple[str]]:
    # English
    COMMON_ENDS = {
        2: ["ed", "er", "ly", "ty", "ry", "al", "el", "an", "en", "or", "ic", "se"],
        3: ["ing", "ist", "ate", "ous", "ent", "ect", "eur", "ess", "ery"],
        4: ["able", "ment", "tion", "tive", "ship", "ness", "sion", "ance"]
    }
    return _parse_unk(corpus, COMMON_ENDS)


def unk_tagging_fr(corpus: list[Tuple[str]]) -> list[Tuple[str]]:
    # French
    COMMON_ENDS = {
        2: ["er", "nt", "re", "on", "ur", "ir"],
        3: ["ion", "ire", "que", "ant", "ent", "ser", "eur"],
        4: ["ment", "tion", "aire", "sion", "ance"]
    }
    return _parse_unk(corpus, COMMON_ENDS)


def unk_tagging_uk(corpus: list[Tuple[str]]) -> list[Tuple[str]]:
    # Ukranian
    COMMON_ENDS = {
        2: ["ий", "ти", "ка", "ок", "ія", "ик", "на", "ць"],
        3: ["ися", "ник", "сть", "ння"],
        4: []
    }
    return _parse_unk(corpus, COMMON_ENDS)


def unk_tagging_pl(corpus: list[Tuple[str]]) -> list[Tuple[str]]:
    # Polish
    COMMON_ENDS = {
        2: ["ać", "ąć", "ić", "eć", "wy", "ść", "yć", "ia", "ki", "ka", "wo"],
        3: ["cja", "nie"],
        4: []
    }
    return _parse_unk(corpus, COMMON_ENDS)


def unk_tagging_ko(corpus: list[Tuple[str]]) -> list[Tuple[str]]:
    # Korean
    COMMON_ENDS = {
        2: ["+의", "+는", "+을", "+에", "+ㄴ", "+이", "+다", "+고", "+은", "+를"],
        3: ["+으로", "+에서", "+ㄴ다", "+라는", "+이나"],
        4: ["+ㅂ니다"]
    }
    return _parse_unk(corpus, COMMON_ENDS)


def _parse_unk(corpus, ends) -> list:
    hapaxes_map = {i: "UNK" for i in FreqDist(
        w for (t, w) in corpus).hapaxes()}
    hpx = len(hapaxes_map.keys())
    print("UNK: replacing %d hapaxes (%.2f%% of corpus)" % (
        hpx, hpx/len(corpus)))
    for k in hapaxes_map.keys():
        # email, e.g. me@sta.ac.uk
        if re.match("([\w]+[\.-]?)+@([\w]+\.[A-Za-z]+)+", k):
            hapaxes_map[k] = "UNK@UNK"
        # acronym, r.g NASA, FBI
        elif re.match("[A-Z]{2,6}", k):
            hapaxes_map[k] = "ACRONYM"
        # numeric, e.g 1871, 2nd
        elif re.match("\d+(st|nd|rd|th)?", k):
            hapaxes_map[k] = "NUMERIC"
        # names, e.g. Johnson, Warwick
        elif re.match("(([A-Z][a-z]+)-?)?[A-Z][a-z]", k):
            hapaxes_map[k] = "PROPER_NAME"
        else:
            for i in SUFFIX_LEN:
                if k[-i:] in ends[i]:
                    hapaxes_map[k] = _replace_stem(k, i)
                    break
    return [(t, _map_word(w, hapaxes_map)) for (t, w) in corpus], hapaxes_map


def _replace_stem(word: str, end: int) -> str:
    return word[-end:]


def _map_word(word: str, map: dict) -> str:
    if word in map.keys():
        return map[word]
    return word
