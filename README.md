# V N G O L

Meta Repository. There are sub-projects which are bundled to the ungol package.

## Setup

Check out this repository. You should now have this folder structure:

```
└── ungol
    ├── eval
    ├── common
    ├── models
    └── similarity
```

To install all dependencies and install the whole module run `make
setup`. To create a wheel run `make dist`.


## Full Run

In the following a full run through the whole system from training
binary embeddings up to an evaluation with an information retrieval
data set is explained.

### Training Binary Embeddings

The `ungol.models` module is concerned with the production of binary
embedding codes via `embcompr.py`. It offers the infrastructure for
training, an implementation of a neural compressor model and the
routines for persisting the codes and models.

First enter the project, create the output directory and inspect the
example configuration and possible options:

``` shell
cd ../models
mkdir opt
less conf/embcompr.example.conf
./ungol/models/embcompr.py --help
```

You need to provide some embeddings. The embedding providers are
defined in `ungol.common.embed` and you can implement your own if any
is missing for your specific format or service. Now create your own
configuration or use an existing one and train a model (from now on,
as an example, a binary model with 256 bit codes is used):

``` shell
./ungol/models/embcompr.py conf/embcompr.binary.conf
```

Various files and statistics are produced and persisted transparantly
by this program. Simply explore the directory given by `folder`
directive of the `statistics` configuration section (in this case
`opt/experiments/current/256x2/`). Much more detailed information can
be found in the `README.md` of `ungol.models`.


### Producing a binary code file

A trained model can now be used to produce a code file for later use
by the `ungol index`. The `embcodr.py` program in `ungol.models` is
concerned with handling these code files. Consider the model produced
by `conf/embcodr.binary.conf` is persisted to
`opt/experiments/current/256x2/compressor/model-2000.torch`:

``` shell
./ungol/models/embcodr.py --help
exp=opt/experiments/current/256x2
./ungol/models/embcodr.py --binary $exp/compressor/model-2000.torch 256x2
```

Now this writes three files:

```
codemap.model-2000.bin
codemap.model-2000.h5
codemap.model-2000.txt
```

The `.bin` file is a the main binary exchange format with minimal
memory footprint. The `.h5` file saves the data in the 1-byte-array
format used internally (and thus consumes much more memory). The
`.txt` file is a human readable version of the data.


### Create k-NN files

Optionally, k-NN files can be created for different distance measures
such as Euclidean, Cosine or Hamming. These are used for analysis in
the notebooks or optionally for index creation. As such it is
described here how to produce such a file for the Hamming distance
(this example uses the h5py embedding provider as defined in the
configuration):

``` shell
./ungol/models/analyze.py --help
exp=opt/experiments/current/256x2
ungol/models/analyze.py nn-hamming $exp/codemap.model-2000.h5 $exp/hamming.h5
```

To create such distance files for other distance measures in
continuous space, simply provide the necessary `--embed-*` command
line arguments (see `--help`) and choose from `nn-manhattan`,
`nn-euclidean` and `nn-cosine`.


### Build Indexes

Now everything is prepared for building an index. The module
`ungol.index.setup` is offering the necessary infrastructure to create
instances of `ungol.index.index.Index` used for calculating different
document distance metrics.

#### SDA/FRA/SPIEGEL

The `CLEF 2003 Ad-Hoc Monolingual`-task works with a document corpus
of around 300k German news articles. These articles must now be added
to a new index. The `eval` project and its `ungol.eval.retrieval`
module are used to do so. Reading in the corpus data is handled by
`dumpr` which reads in xml files. If you do not wish to use this
approach have a look at
`ungol.eval.retrieval.setup.do_ungol_setup_articles` - it is quite
straight forward to implement your own reader. You need a pickled
dictionary of the word embedding indexes and a `codemap` as described
in the former section.


``` shell
./ungol/retrieval/setup.py ungol-setup-articles \
    --vocabulary path/to/vocab.pickle \
    --ungol-codemap path/to/codemap.h5 \
    --files path/to/*.xml \
    --processes 7 \
    --out opt/indexes/fasttext.de-256x2.index.pickle
```

Additionally, for the evaluation, the query documents also need to be
added to the index:

``` shell
ungol/retrieval/setup.py ungol-setup-topics \
    --files data/CLEF/truth/CLEF2003_ah-mono-de_topics.xml \
    --out opt/indexes/fasttext.de-256x2.index.pickle
```


### Run an Evaluation

Now much information is passed by using the command line arguments (I
may implement configfile as in `ungol.models` but there are more
important things to do atm.). I usually aggregate common command line
arguments in variables (see `eval/dot_me_honey.fish`.

To run an evaluation on a randomly sampled but fixed dataset (created
by `eval/notes/dataset.ipynb`) run:

``` shell
./ungol/retrieval/evaluate.py \
    data/CLEF/truth/CLEF2003_ah-mono-de.txt \     # ground truth
    opt/current/ \                                # directory to write results to
    --ungol \                                     # evaluate ungol
    --ungol-index opt/indexes/fasttext.de-256.index.pickle \
    --ungol-fn rhwmd                              # scoring function
    -k 250                                        # retrieve a selection of 250 documents
    --dataset opt/datasets/180921.10000.1.pickle  # pre-selection
```



### Additional Information

#### Logging

The whole library uses `ungol.common.logger` which in turn uses the
standard python logging facility. It checks for two possible logging
configurations:

1. If (based on the cwd of execution) a file `conf/logging.conf`
   exists it is used.
2. If the environment variable `UNGOL_LOG` is set to a `*.conf` file
   then this file is used. It overwrites the `conf/logging.conf` if it
   exists.
