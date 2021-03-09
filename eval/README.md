# VNGOL ES

Main repository for the IR experiment.

This repository contains scripts for setting up elasticsearch for
different deployment scenarios, feeding example data to play around
and python notebooks exploring the python client.


## Installation

```
mkdir lib
pip install -e requirements.txt
```


### Retrieval

```
cd lib
git clone https://github.com/dtuggener/CharSplit
```


### SentEval

```
pushd lib
git clone https://github.com/facebookresearch/SentEval
pushd data/downstream
./get_transfer_data.bash  # requires about 1.3G of space
popd
```



# Legacy

To be checked again and then removed.

## Experimental Results

Everything's measured in mean average precision. Each following
database contains the changes to the previous ones.

| Database                   | Comment             | min | max | adp-small | adp-big |
|----------------------------|---------------------|-----|-----|-----------|---------|
| `180910_ungol.db.pickle`   | _The Deprived_      | 20  | 23  | 23        | 20      |
| `180914_ungol.db.pickle`   | + Compound Splitter | 25  | 27  | 30        | 25      |
| `180914-2_ungol.db.pickle` | + Unknown Words     | 26  | 30  | 30        | 26      |
| `180915_ungol.db.pickle`   | + NR-IDF            | 16  | 30  | 30        | 15      |
| `180915-2_ungol.db.pickle` | - Stopwords         | 16  | 27  | 27        | 15      |
| `180917_ungol.db.pickle`   | + IDF               | 14  | 34  | 35        | 13      |
| `180917_ungol.db.pickle`   | + IDF               | 20  | 35  | 35        | 20      |

1. `180910_ungol.db.pickle`
   - First database. Very rudimentary preprocessing - tokens were only
   stripped and lowercased.
   - https://git.ramlimit.de/deepca/ungol-es/tree/c1a04c5dcbdd700c9c7ab89935cc665e6a78cf86
   - https://git.ramlimit.de/deepca/ungol-wmd/tree/b04b08ab96f4f53ae7a1ed436988bde6a43e6621
2. `180914_ungol.db.pickle`
   - More advanced preprocessing. Stripping tokens from
   non-alphanumeric characters, using dtuggener/CharSplit compound
   splitter.
   - https://git.ramlimit.de/deepca/ungol-es/tree/ce09b253c3fecaae16ebd7cb327cee62970db42f
   - https://git.ramlimit.de/deepca/ungol-wmd/tree/7474b3495208fb98380a09387262af7cecd54148
3. `180915_ungol.db.pickle`
   - Added termfreqs and docfreqs to the docrefs. Used for calculating
   the idf in wmd.dist.
   - https://git.ramlimit.de/deepca/ungol-es/tree/2a554a2617b0ad49b92793977dbff8b4bcfbd982
   - https://git.ramlimit.de/deepca/ungol-wmd/tree/31ea4b99eb35853e66610ecb3cbabafd6b5a355d
4. `180915-2_ungol.db.pickle`
   - Same as 180915 but not using any stopwords.
   - https://git.ramlimit.de/deepca/ungol-es/tree/2a554a2617b0ad49b92793977dbff8b4bcfbd982
   - https://git.ramlimit.de/deepca/ungol-wmd/tree/31ea4b99eb35853e66610ecb3cbabafd6b5a355d
5. `180917_ungol.db.pickle`
   - Added Doc.freq and use Doc.cnt for absolute values. Fixed
     DocReferences.termfreq mapping. Calculating the idf such as God
     intended it to be calculated.
   - https://git.ramlimit.de/deepca/ungol-es/tree/b84137a428343e467db27ebf482c577e1dcbca0f
   - https://git.ramlimit.de/deepca/ungol-wmd/tree/ed1b3604c2650a282b5d818149174df5a992e04f
6. `180919_ungol.db.pickle`
   - ?
   - https://git.ramlimit.de/deepca/ungol-es/tree/cd90f4f257c956dbddea479bd5a628535985247a
   - https://git.ramlimit.de/deepca/ungol-wmd/tree/8bc1e6118f44d2929e8c27a58ca3f12812c8031a
7. `180921_ungol.db.pickle`
   - Added db.avg_doclen and db.valid
   - https://git.ramlimit.de/deepca/ungol-es/tree/c0c9394b45bb89a7815e2171c58e8076b3178cf8
   - https://git.ramlimit.de/deepca/ungol-wmd/tree/ce9a988b0916f1b4b66f0682334fbb5f9d6ee0f7


### Python

```
pip install -r requirements.txt
ipython kernel install --user --name=ungol-data
mkdir lib
cd lib
git clone https://github.com/dtuggener/CharSplit
```

Also, other ungol modules and dumpr need to be installed:

```shell
mv to/other/project
pip install -r requirements.txt
pip install -e .
```

### Local Elastic Installation

The `es.sh` script handles a local elasticsearch installation. If you
wish to change the target directory (opt/install), change
`INSTALL_DIR` in the file. The script offers the following options:

* `es.sh install` installs Elasticsearch and Kibana to
  opt/install. Make sure that opt/install exists.
* `es.sh run elastic` starts the local one-node elasticsearch cluster
* `es.sh run kibana` starts the local kibana web application


### Elastic Docker

Consult this for setting production mode kernel parameters:

* https://www.elastic.co/guide/en/elasticsearch/reference/current/docker.html

``` shell
docker pull docker.elastic.co/elasticsearch/elasticsearch:6.2.4
docker run -p 9200:9200 -p 9300:9300 -e 'discovery.type=single-node' docker.elastic.co/elasticsearch/elasticsearch:6.2.4
```

* cluster Status: `curl http://127.0.0.1:9200/_cat/health`
* pass Variables: `docker run -e "cluster.name=theclustername" ...`
* ulimit adjustments: `docker run --ulimit nofile=65536:65536 ...`
* java memory adjustments: `docker run -e ES_JAVA_OPTS="-Xms16g -Xmx16g" ...`

In general:

* Use bind mounts for `/usr/share/elasticsearch/data/` in production for persistent data


### Full Run Notes

```
for dataset in opt/datasets/*
    echo
    echo BM25
    echo $dataset $fn
    ungol/experiments/evaluate.py $ungol_argv --ungol-db $db --ungol-fn bm25 --ungol-verbose --dataset $dataset

    for strat in min max adaptive-small adaptive-big sum
        echo
        echo RHWMD $strat
        ungol/experiments/evaluate.py $ungol_argv --ungol-db $db --ungol-fn rhwmd --ungol-verbose --dataset $dataset --ungol-strategy $strat
    end
end
```
