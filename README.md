# Fork of Andrej Karpathy's makemore

This code is a fork of Andrej Karpathy's makemore project, aimed at fine-tuning hyperparameters for makemore and experimenting with datasets beyond names.txt.

## Intent

Initially, the goal was to apply makemore to various datasets other than names.txt.

## Modifications

The primary modifications to makemore include an option for early stopping. When the code runs and the loss ceases to decrease further, it checks every 500 steps where the models usually save. If the loss hasn't decreased, a watchdog count increases. The command-line parameter for early stopping determines how many attempts to wait for the loss to decrease before quitting. For example, if set to one, it goes through that iteration once and then stops; if set to ten, it goes through ten times. This is Change #1.

Another modification involves setting a target threshold for loss. While determining the optimal batch size, I observed how the code's runtime changes with different batch sizes. By varying the batch size, makemore can reach a lower loss more quickly or slowly. The idea is to run a script, set a target, and then identify which batch sizes, fed as command-line arguments to makemore, produce the shortest run times.

Also a hex tag for easy seraching when using a script to grid search hyperparameters

Finally, output a CSV file with hex tag,step #, train and test loss...

sort -nk4 -t, -r makemore_log.csv | tail
0x324fef,500,0.47,0.501
0x119fbc,500,0.469,0.5
0x5ba887,500,0.469,0.499
0x3254ff,500,0.467,0.499
0xa5510a,500,0.468,0.498
0xfca52,500,0.469,0.497
0xbf86d,500,0.455,0.496
0xa5362b,500,0.455,0.496
0x9d98df,500,0.451,0.495
0x78a5b6,500,0.451,0.495

## Scripts

There are several scripts, one for tuning. This script varies the hyperparameters using a Bash shell script and passes the parameters as command-line arguments to makemore. The intention was not to alter the makemore code significantly but to run it externally using batch shell scripts. The concept of tagging is important here. Specific outputs produced by makemore are tagged and can be redirected to a file, including the dictionary of hyperparameters. The best loss is also tagged, which is saved after every 500 steps if it's better than previous validation losses. This tagging helps when running multiple iterations of makemore and searching for early stopping or the best loss. Tags are random hexadecimal numbers, usually 24 bits to minimize collision probability.
Others mainly are for running the makemore.py with specific paramters, input files and output directories.

## Initialization Lottery

Another experiment involved what I call an "initialization lottery." I generated random seeds using a shell script to run makemore with different seed values. This experiment revealed variations in loss performance over time, suggesting that certain seed values are luckier than others. All this information is documented here.

## Seed Runs

### Less data and more data fed in

I also experimented with different datasets. One example was generating addition examples using a Python script called `addition.py`. This script creates 10,000 examples of addition problems, restricting itself to numbers from zero to nine without using operators. The resulting dataset consists of input numbers and their corresponding sums. This dataset is smaller than `names.txt` and can lead to overfitting quickly.

821294
033437
8812100
064248
393372

### Sqrt

Another experiment involved calculating square roots. A Python helper function generates square root data for numbers from 1 to 1000. The output includes the integer number, a colon, a space, and the square root to four decimal places. Makemore, when trained on this data, performs well and can generalize to unseen examples.

978: 31.3169
970: 31.149:
682: 26.5543
447: 21.2038
682: 26.5543

### English Words Dataset (about 10x larger than names)

I also experimented with a significantly larger dataset—approximately ten times larger than `names.txt`. I combined three sources of English words, resulting in a larger dataset. Training on this dataset yields interesting English-like words and demonstrates longer periods of generating new output before memorizing. It's a fun dataset to play with.

comprecise
arrogenry
ckikeke
revotial
sinharmic

### Final 10M Quotes File

Lastly, there's a 10-megabyte file of quotes with sentences that contain punctuation, capitalization, and a variety of characters. This serves as a challenging test for makemore. Running this dataset may consume more memory and time compared to smaller datasets, and it's an interesting experiment to see how well makemore can handle it.

* I love spending the whole decades, as a couchy.
* He's what great political after someone we maguare if one matters.
* It was the newspaed our romantic between duty. In this New Gesoton to Bart I was gotten to make her sad effect with my mom was together, all the Americans that was learning marriage was more he happiness.
* It is eating moral, but one two years against the effort American proceeds looks back to a master once.

## Conclusion

This fork of makemore offers various modifications and experiments to explore its capabilities beyond its initial use case. Feel free to explore and enjoy the code, and hopefully, you'll learn something new. Happy experimenting!
Plus a few sidequests.

## Large README
There is a large verbose-readme.pdf for much more details.

## Acknowledgements
Andrej Karpathy ( https://github.com/karpathy )for paving the way to the frontier of AI with good tutorials and code to learn from.
Gwern Branwen ( https://gwern.net/ ) and his TWDNE Project ( https://gwern.net/twdne ), which helped me get motivated again with ML projects, especially the quotes training as a warmup for something bigger in the future.
Lex Fridman ( https://lexfridman.com/ )for educational podcasts that keep me in the loop on AI and more even when I am 'not feelin it' with code.

# Original README ...
# makemore

makemore takes one text file as input, where each line is assumed to be one training thing, and generates more things like it. Under the hood, it is an autoregressive character-level language model, with a wide choice of models from bigrams all the way to a Transformer (exactly as seen in GPT). For example, we can feed it a database of names, and makemore will generate cool baby name ideas that all sound name-like, but are not already existing names. Or if we feed it a database of company names then we can generate new ideas for a name of a company. Or we can just feed it valid scrabble words and generate english-like babble.

This is not meant to be too heavyweight library with a billion switches and knobs. It is one hackable file, and is mostly intended for educational purposes. [PyTorch](https://pytorch.org) is the only requirement.

Current implementation follows a few key papers:

- Bigram (one character predicts the next one with a lookup table of counts)
- MLP, following [Bengio et al. 2003](https://www.jmlr.org/papers/volume3/bengio03a/bengio03a.pdf)
- CNN, following [DeepMind WaveNet 2016](https://arxiv.org/abs/1609.03499) (in progress...)
- RNN, following [Mikolov et al. 2010](https://www.fit.vutbr.cz/research/groups/speech/publi/2010/mikolov_interspeech2010_IS100722.pdf)
- LSTM, following [Graves et al. 2014](https://arxiv.org/abs/1308.0850)
- GRU, following [Kyunghyun Cho et al. 2014](https://arxiv.org/abs/1409.1259)
- Transformer, following [Vaswani et al. 2017](https://arxiv.org/abs/1706.03762)

### Usage

The included `names.txt` dataset, as an example, has the most common 32K names takes from [ssa.gov](https://www.ssa.gov/oact/babynames/) for the year 2018. It looks like:

```
emma
olivia
ava
isabella
sophia
charlotte
...
```

Let's point the script at it:

```bash
$ python makemore.py -i names.txt -o names
```

Training progress and logs and model will all be saved to the working directory `names`. The default model is a super tiny 200K param transformer; Many more training configurations are available - see the argparse and read the code. Training does not require any special hardware, it runs on my Macbook Air and will run on anything else, but if you have a GPU then training will fly faster. As training progresses the script will print some samples throughout. However, if you'd like to sample manually, you can use the `--sample-only` flag, e.g. in a separate terminal do:

```bash
$ python makemore.py -i names.txt -o names --sample-only
```

This will load the best model so far and print more samples on demand. Here are some unique baby names that get eventually generated from current default settings (test logprob of ~1.92, though much lower logprobs are achievable with some hyperparameter tuning):

```
dontell
khylum
camatena
aeriline
najlah
sherrith
ryel
irmi
taislee
mortaz
akarli
maxfelynn
biolett
zendy
laisa
halliliana
goralynn
brodynn
romima
chiyomin
loghlyn
melichae
mahmed
irot
helicha
besdy
ebokun
lucianno
```

Have fun!

### License

MIT
