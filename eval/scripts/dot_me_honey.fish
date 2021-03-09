# . dot_me_honey.fish
# for quick setup, expose some variables
# afterwards, invoke e.g.: ungol/experiments/evaluate.py $eval_argv --elastic
#

# dirs
set -gx opt opt
set -gx exp $opt/tmp
set -gx src $opt/src

# produced by ungol
set -gx vocab $src/fasttext.de.vocab.pickle
set -gx codemap $src/codemap.bin

set -gx es_endpoint --elastic-endpoint localhost:9200

# clef data
set -gx task $src/task.clef2003.txt
set -gx topics $src/topics.clef2003.xml
set -gx articles \
  $src/dump/sda-94/sda-94.full.xml \
  $src/dump/sda-95/sda-95.full.xml \
  $src/dump/spiegel-9495/spiegel-9495.full.xml \
  $src/dump/frankfurter-rundschau-9495/frankfurter-rundschau-9495.full.xml

# other
set -gx stopwords $src/stopwords/*txt
set -gx embeddings $src/fasttext.de.pickle
set -gx wmd_embeddings $src/fasttext.de.bin

#
# aggregations
#

# for evaluate.py
set -gx argv_eval $task $exp -k 250
set -gx argv_reranking --preselect 250

set -gx argv_es $argv_eval $es_endpoint
set -gx argv_un $argv_eval --ungol
set -gx argv_un_reranking $argv_eval $es_endpoint  --ungol-reranking $argv_reranking

set -gx argv_wmd $argv_eval --wmd --wmd-embeddings $wmd_embeddings
set -gx argv_wmd_reranking $argv_eval $es_endpoint --wmd-reranking $argv_reranking
set -gx argv_wmd_relaxed $argv_eval --wmd-relaxed --wmd-embeddings $embeddings
set -gx argv_wmd_relaxed_reranking $argv_eval --wmd-relax-reranking --wmd-embeddings $embeddings

# for setup.py setup-*-ungol
set -gx setup_argv --vocabulary $vocab --ungol-codemap $codemap
