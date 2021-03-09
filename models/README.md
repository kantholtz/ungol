# UNGOL MODELS

**Contents**
* [Installation](#installation)
* [Interface](#interface)
  * [Embedding Compressor](#embedding-compressor)
  * [Produce Codes](#produce-codes)
  * [Analyze](#analyze)
  * [Create k-NN files](#create-k-nn-files)
  * [Create correlation files](#create-correlation-files)
* [Data](#data)
  * [Notes Regarding h5py](#notes-regarding-h5py)
  * [Who processes data](#who-processes-data)
  * [Generated Images](#generated-images)
  * [Produced Data](#produced-data)
    * [opt/correlation](#opt/correlation)
    * [opt/embed](#opt/embed)
    * [opt/experiments](#opt/experiments)
    * [opt/neighbours](#opt/neighbours)
  * [Sentence Embeddings](#sentence-embeddings)


# Installation

Please see the readme of https://github.com/ungol-nlp/meta/blob/master/README.md


# Interface

This section lists common operations done with the executables and
gives some examples:


## Embedding Compressor
`ungol/models/embcompr`

Here the compressor model is trained. All configuration is done in a
separate configuration file: see `conf/embcompr.example.conf` for more
details.

```
usage: embcompr.py [-h] [--no-gpu] [--baseline] conf

positional arguments:
  conf        configuration file

optional arguments:
  -h, --help  show this help message and exit
  --no-gpu    disable training on GPU
  --baseline  use the baseline encoder
```

The options are quite self explanatory. The usual invocation simply
looks like this:

``` shell
ungol/models/embcompr.py conf/embcompr.golden.conf
```


## Produce Codes
`ungol/models/embcodr`

This program loads a trained model and produces files containing the
codes based on a vocabulary. It also offers an interface to transform
the encoder output for other applications when `ungol` is installed as
a python module.


``` shell
exp=opt/experiments/binary/fasttext.de-256x2
ungol/models/embcodr.py $exp/compressor/model-2000.torch fasttext.de-256x2 [--binary]
```

In case of `--binary` this will create three files:

```
...
├── codemap.model-500.bin
├── codemap.model-500.h5
├── codemap.model-500.txt
...
```

## Analyze
`ungol/models/analyze`

All statistical data produced while training is transformed here. But
mostly operations on the raw embedding spaces are implemented
herein. The help itself is not so useful atm:

```
 WELCOME TO THE ANALYZER

usage: analyze.py [-h] cmd [argv [argv ...]]

positional arguments:
  cmd
  argv

optional arguments:
  -h, --help  show this help message and exit
```

A thorough explanation of the different `cmd` options is given in the
file itself. Some short examples:


### Create k-NN files

To create files with (currently) at most 4k nearest neighbours for
each word of the vocabulary, the nn-* command family is used. This,
for example, creates a cosine distance file for glove:

``` shell
ungol/models/analyze.py nn-cosine --embed-provider ...
```

For data produced by the model while training use:

``` shell
exp=opt/experiments/binary/fasttext.de-256x2
ungol/models/analyze.py nn-hamming $exp/codemap.model-2000.h5 $exp/hamming.h5
```

For data produced by embcodr:

``` shell
exp=opt/experiments/binary/fasttext.de-256x2

```


### Create correlation files

To investigate to which degree different distance metrics correlate, a
file containing the common nearest neighbours of two different
embedding spaces (in this work usually linear space with cosine
distances and a hamming space with embedded codes) are created with
the `correlate` cmd. Provide as last values the different k that are
of intereset. The computation of lower k is a subset of higher k, so
you should provide multiple.

``` shell
vocab=opt/embeddings/glove/vocabulary.pickle
ref=opt/neighbours/glove-cosine.h5
cmp=opt/experiments/binary/glove-256x2/codes.h5
out=opt/correlation/cosine-binary_256x2/
mkdir -p $out
ungol/models/analyze.py correlate $ref $cmp $vocab $out 10 100 1000'
```


# Data

Description of internally used data formats

## Notes regarding h5py

Data is stored either as pandas data frames or raw numpy arrays. Most
data is processed chunk wise as it will not fit into memory
(development machine had 23GB of RAM - initially at least ;) ).

Regarding the chunking implemented into hdf5:

> Also be aware that if a dataset is read by whole chunks and there is
> no need to access the chunks more than once on the disk, the chunk
> cache is not needed and can be set to 0 if there is a shortage of
> memory.

Since the order of reading is always sequential, chunks are generally
not used but slices from large continuous arrays are copied into
memory and released shortly after computation.


## Who processes data

 Data transformation takes place inside `ungol/models/stats.py` while
training and `ungol/models/analyze.py` afterwards (for data from the raw
embedding space or data produced by `stats.py`). The migrations for
legacy data to new data formats can be found in `ungol/models/migrations.py`.

### Generated images

Images can usually be found inside
`opt/experiments/<SERIES>/<NAME>/images/` both as png and svg. The
following table gives an overview which code produced the
corresponding image. For a nice overview of the generated images,
produce `opt/experiments/<SERIES>/<EMBEDDING>.html` files with
`notes/overview.ipynb`.


| image name                 | responsible code       | comment                                    |
|----------------------------|------------------------|--------------------------------------------|
| loss                       | `notes/training.ipynb` | training/validation loss curve             |
| loss-training              | `notes/training.ipynb` | training curve including interval          |
| loss-validation            | `notes/training.ipynb` | validation curve including interval        |
| encoder-activation-train_* | `notes/training.ipynb` | pie-plot with three classes of activations |
| encoder-activation-valid_* | `notes/training.ipynb` | bar-plot with histogram of code selection  |
| codebooks-*                | `notes/model.ipynb`    | t-SNE visualization of codebook vectors    |
| norms-*                    | `notes/model.ipynb`    | bar-plot with distances and vector norms   |


## Produced Data

Found in `opt` elsewhere outside this repository. This section
describes the conventions used for storing data which is also
reflected when you download pre-trained models or statistical data.

FIXME: add download link

### opt/correlation

Produced by `ungol/models/analyze.py correlate ...`

All subfolders in `opt/correlation/` have the following naming
convention: `$ref-$cmp` where `$ref` and `$cmp` are the respective
distance metrics. For the hamming distances produced by the models,
usually `$cmp` is set to the corresponding experiment name.

Correlation files are written as `.h5` files and have the following
structure:

```
    N: vocabulary size
    k: the k in k-NN

    "ref/common" shape (k, N) with words (str) denoting columns and
                 distances to their common nearest neighbours in
                 cmp/common

    "cmp/common" shape (k, N) with words (str) denoting columns and
                 distances to their common nearest neighbours in
                 ref/common

    DataFrames contain nan values (and thus must be float).

```


### opt/embed

This folder contains raw embedding files used by
`ungol.common.embed`. See `ungol.common` for more details.


### opt/experiments

Produced by `ungol/models/embcompr.py ...` and `ungol/models/embclustr.py ...`

Experiments conducted by using the neural compressor or code
cluster. For details regarding the models see their corresponding code
and documentation. Usually the following files are produced:

For each `opt/experiments/<SERIES>/<EXPERIMENT>/`

* `compressor/`
  * `compressor-XXXX.torch` : Model states of the compressor
    to re-create models for a specific epoch.
  * Supplemental files for the `*.torch` files to re-create models
    * `<MODEL>-dimensions.json`
    * `<MODEL>-dimensions.json`
* `<MODEL>.conf` Original training configuration
* `raw/*` Losses, gradients, codes etc. produced while training. This
  is controlled by the statistics settings in the configuration. This
  directory is obsolete after switching to hdf5 throughout.

**Data Container**

* `codes.npz` produced by `embcodr.py`
* `codes.txt` produced by `embcodr.py`


**hamming.h5** Hamming Distance of codes (formerly `hamming/hamming.*.npz`)

```

  N: vocabulary size
  M: Code components
  R: retained nearest neighbours (4k so far for all data)

  Datasets:
    "dists":     shape (N, R) float
    "mapping":   shape (N, R) float
    "histogram"  shape (M, )

 This data contains nan values (thus all dtypes are float).

```


**losses.h5** Average, batch-wise loss values (formerly `raw/losses-(valid|train)-*.npz`)

```
  B: batch count (total // batch_size)
  T: epoch count

  Datasets:
    "train": shape float (T, B)
    "valid": shape float (T, B)

  Data does not contain nan values.

```


**gradients.h5** Average, batch-wise gradient values (formerly `raw/grad_(enc|dec)-train-*.npz`)

```
  B: batch count (total // batch_size)
  T: epoch count

  Datasets:
    "encoder": shape (T, B) float [optional]
    "decoder": shape (T, B) float [optional]

  Data does not contain nan values.

```


**codes.h5**

FIXME: #24

```
  N: vocabulary size
  M: code components
  K: codelength per component (for one-hot encoding)

  <EPOCH>: integer of selected epoch (remember to int(x) for sorting)

  Groups:
    "train/"
      "train/<EPOCH>/": datasets for EPOCH

      Datasets:
        "raw": shape (N, M, K) float
        "histogram": shape (bins, ) integer
          - attr "bin_edges": edges of the histogram bins
        "entropy": shape (M, ) float
          - attr "mean": mean entropy over all codebooks

    "valid/"
      "valid/<EPOCH>/": datasets for EPOCH

      Datasets:
          "raw":     shape (N, M) integer
          "counts":  shape (M, K) integer histogram per codebook
          "entropy": shape (M, ) float
            - attr "mean": mean entropy over all codebooks

  Data does not contain nan values.

```


### opt/neighbours

Files produced by the `ungol/models/analyze nn-* ...` command family. Legacy
files were 100 chunks of `.pickle` files. Newer versions switched to
h5. The `.h5` files contain the following data:

```
  N: vocabulary size
  k: the k in k-NN
  b: bin count of the histogram

  Datasets:

  "dists":     shape (N, k) with float values
  "mapping":   shape (N, k) with integer values
  "histogram": shape (b, )
    - attr: "bin_edges"
  "histogram_<k>-NN"  (written by notes/neighbours.ipynb)
    - attr: "bin_edges"

  Data does not contain nan values.

```


## Sentence Embeddings
`ungol/sentemb`

Used to create sentence embeddings for training with embcompr. If you
have a `dumpr` xml file, the first step is to produce a file
containing sentences (one per line):

```
ungol/sentemb/prep.py create-file --dumpr /path/to/xml --out opt/sentences.txt
```

Now if you have a text file containing one sentence per line you can
use this file to tokenize the sentences, build a vocabulary and count
the word frequencies:

```
ungol/sentemb/prep.py from-file --file opt/sentences.txt --out opt/ --processes 4
```

This creates four files:

```
└── opt
    ├── counts.pickle
    ├── sentences.arrs
    ├── tokens.txt
    └── vocab.pickle
    ...
```

The files `tokens.txt` contain tokens and `sentences.arrs` contains
the token indexes. The `vocab` and `counts` files are two pickled
dicts containing the `index -> string` and `index -> term count`
mappings.

Optionally, a second pass can be done on the data pruning all words
which do not occur often enough (the following command would remove all words
which have a term frequency of under 10):

```
ungol/sentemb/prep.py prune --folder opt/ --threshold 10
```

Now a training data set can be created using some word embeddings:

```
ungol/sentemb/training.py create \
  --embed-provider h5py --embed-file-name path/to/embed.h5 --embed-vocabulary path/to/vocab.pickle \
  --folder opt/ [--prefix pruned-10] --out opt/ --redux mbow
```

This produces a sentence file where the line number of each sentence
corresponds to the index of the embeddings in the .h5 file. This .h5
file can now be used with the h5py embed provider for training
embcompr.


### Create sentence embeddings and analyze their NN

FIXME: outdated information; update this using the current `dotme.fish`.

A common use case is to examine whether a method for combining
sentence embeddings produces vectors of high quality;
i.e. semantically close sentences should have a low spatial
distance. The methods explored can be found in `ungol.sentemb.redux`.

The following reduction methods are offered (you are free plug your
own there):

* `redux.BoW (--redux bow)`: sum of all vectors
* `redux.MBoW (--redux mbow)`: mean of all vectors
* `redux.WMBoW (--redux wmbow)`: weighted mean (weight is 1/p(w))
* `redux.SIF (--redux sif)`: tough to beat baseline implementation

A typical workflow looks like this (using GloVe embeddings, the SICK
data set and SIF as reduction method):

```
set -l embed_glove --embed-provider h5py --embed-file-name opt/embed/glove-840b.2m.300d.h5 --embed-vocabulary opt/embed/glove-840b.2m.300d.vocab.pickle
set -l sick --folder opt/bow/sick/src --out opt/bow/sick

ungol/sentemb/training.py create $embed_glove $sick --processes 10 --redux sif --sif-alpha 0.0001
ungol/sentemb/training.py vocab --file opt/bow/sick/sif.sentences.txt

set -l embed_sif --embed-provider h5py --embed-file opt/bow/sick/sif.embedding.h5 --embed-vocabulary opt/bow/sick/sif.vocab.pickle
ungol/models/analyze.py nn-cosine opt/bow/sick/sif.cosine-dist.h5 $embed_sif
```

This can be used for an exemplary analysis using the `notes/knn.ipynb`
notebook for example.


# Handy Processing Notes

## Wikipedia Sentence Embedding

Parameters used for a 64 core machine:

```fish
set -x redux mbow
set -x dataset enwiki_1m
set -x bits -1

. dotme.fish


function prep -a lower upper
    echo prepping $lower $upper
    for i in (seq $lower $upper)
        mkdir -p opt/data/$dataset/control/$i
        set -l in opt/data/$dataset/src/raw/$dataset-$i.sentences.txt
        set -l out opt/data/$dataset/control/$i/
        ungol/sentemb/prep.py from-file --file $in --out $out --processes 5 >/dev/null 2>&1 &
        echo dispatched $i
    end
end

function vocab -a lower upper
    echo creating vocab $lower $upper with redux: $redux
    for i in (seq $lower $upper)
        set -l folder opt/data/$dataset/control/$i
        ungol/sentemb/training.py vocab --file $folder/$redux.sentences.txt >/dev/null 2>&1 &
        echo dispatched $i
    end
end


prep 0 10
vocab 0 10


function create -a lower upper
    echo creating $lower $upper with redux: $redux
    for i in (seq $lower $upper)
        set -l folder opt/data/$dataset/control/$i
        ungol/sentemb/training.py create --folder $folder --out $folder  --processes 5 --redux $redux $embed_glove >/dev/null 2>&1 &
        echo dispatched $i
    end
end

set -x redux sent2vec
. dotme.fish

function create -a lower upper
    echo creating $lower $upper with redux: $redux
    for i in (seq $lower $upper)
        set -l folder opt/data/$dataset/control/$i
        ungol/sentemb/training.py create --folder $folder --out $folder  --processes 5 --redux $redux --sent2vec-model opt/lib/sent2vec/wiki_bigrams.bin >/dev/null 2>&1 &
        echo dispatched $i
    end
end

create 0 10
vocab 0 10
```
