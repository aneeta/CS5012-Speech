import argparse
from functools import wraps
from time import time

import pandas as pd

import preprocessing
import hmm


def run_timing(f):
    @wraps(f)
    def wrap(*args, **kw):
        start = time()
        result = f(*args, **kw)
        end = time()
        print('%r took: %2.2f s' % (f.__name__, end-start))
        return result
    return wrap


def get_cli_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-l', '--langs',
        nargs='+',
        default=preprocessing.LANGUAGES,
        help="""Choose a language from ${}.""".format(preprocessing.LANGUAGES))
    parser.add_argument(
        '-u', '--unk',
        action='store_true',
        help="Flag indicating if to use the <UNK> tags.")
    parser.add_argument(
        '-s', '--smoothing',
        default="WB",
        help="""Choose smoothing method from ${}.""".format(hmm.SMOOTHING))
    parser.add_argument(
        '-w', '--warnings',
        action='store_true',
        help="Flag to switch on warnings.")
    parser.add_argument(
        '-p', '--plot',
        default=None,
        type=str,
        help="Figure name preffix for a plot. Plot is the working directory unless other existing directory included in filename.")
    parser.add_argument(
        '-c', '--csv',
        default=None,
        type=str,
        help="Name for csv results file. Saves in the working directory unless other exisitng directory included in filename.")
    args = parser.parse_args()
    return vars(args)


def format_results(results):
    dataframes = []
    for k, v in results.items():
        d = pd.DataFrame.from_dict(v, orient='index').reset_index()
        d['lang'] = k
        dataframes.append(d)
    return pd.concat(dataframes, ignore_index=True)
