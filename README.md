# CS5012-Speech
Practical 1 for CS5012 Language and Computation module at St Andrews.

## Dependencies
The projet also uses the following external Python libraries:
- `conllu` - accessing language corpora
- `nltk` - for NLP functionality (probability distributions)
- `numpy` - for matrix operations
- `sklearn` - for evaluation functions
- `matplotlib`- for plotting

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

```
usage: p1.py [-h] [-l LANGS [LANGS ...]] [-u] [-s SMOOTHING] [-w] [-p PLOT] [-c CSV]

optional arguments:
  -h, --help            show this help message and exit
  -l LANGS [LANGS ...], --langs LANGS [LANGS ...]
                        Choose a language from $['EN', 'FR', 'UK', 'PL', 'KO'].
  -u, --unk             Flag indicating if to use the <UNK> tags.
  -s SMOOTHING, --smoothing SMOOTHING
                        Choose smoothing method from ${'WB': <class 'nltk.probability.WittenBellProbDist'>, 'GT': <class
                        'nltk.probability.SimpleGoodTuringProbDist'>}.
  -w, --warnings        Flag to switch on warnings.
  -p PLOT, --plot PLOT  Figure name preffix for a plot. Plot is the working directory unless other existing directory included in filename.
  -c CSV, --csv CSV     Name for csv results file. Saves in the working directory unless other exisitng directory included in filename.
```

### Troubleshooting

In case of errors, please ensure that all all the dependencies are installed and accessible.

### Results analysis scrips
To easily analyse the results, I wrote a script to automate running different variations of the model.

To run the experiment script
```
bash run_experiments.sh <DIR_NAME>
```
Preferably use `<DIR_NAME>` that does not exist to prevent data getting overriden.
