# CS5012-Speech
Practical 1 for CS5012 Language and Computation module at St Andrews

## Dependencies
The projet also uses the following external Python libraries:
- `conllu` - accessing language corpora
- `nltk` - for NLP functionality (probability distributions)
- `numpy` - for matrix operations
- `sklearn` - for evaluation functions
- `matplotlib`- for visualisations

## How to run

```
python p1.py
```
or

```
python3 p1.py
```

### Command line arguments
The program can be ran with command line arguments to adjust behaviour.
- `-l/--langs` takes a list of languages to run the program for. Defaults to all supported languages.
- `-s/--smoothing` takes a smoothing function. Defaults to WittenBell.
- `-u/--unk` switches on use of UNK tags.

```
usage: p1.py [-h] [-l LANGS [LANGS ...]] [-u] [-s SMOOTHING]

optional arguments:
  -h, --help            show this help message and exit
  -l LANGS [LANGS ...], --langs LANGS [LANGS ...]
                        Choose a language from $['EN', 'FR', 'UK'].
  -u, --unk
  -s SMOOTHING, --smoothing SMOOTHING
                        Choose smoothing method from ${'WB': <class 'nltk.probability.WittenBellProbDist'>, 'GT': <class 'nltk.probability.SimpleGoodTuringProbDist'>}.
```

### Troubleshooting

In case of errors, please ensure that all all the dependencies are installed and accessible.
