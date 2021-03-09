# give me a sign
# dot me baby one more time


# config:

if not set -q redux
    echo 'error: set -x redux <REDUX>'
    echo '  e.g. set -x redux mbow'
    exit 2
end

if not set -q dataset
    echo 'error: set -x dataset <DATASET>'
    echo '  e.g. set -x dataset enwiki_100k'
    exit 2
end


if not set -q bits
    echo 'error: set -x bits <BITS>'
    echo '  e.g. set -x bits 256'
    exit 2
end


# ---

set -x name "$dataset.$redux-$bits"
set -x folders --folder opt/data/$dataset/src/ --out opt/data/$dataset

set -x embed --embed-provider h5py \
  --embed-file opt/data/$dataset/$redux.embedding.h5 \
  --embed-vocabulary opt/data/$dataset/$redux.vocab.pickle

set -x embed_glove --embed-provider h5py \
  --embed-file-name opt/embed/glove-840b.2m.300d.h5 \
  --embed-vocabulary opt/embed/glove-840b.2m.300d.vocab.pickle


# some hard-coded things

set -x sent2vec --sent2vec-model opt/lib/sent2vec/wiki_bigrams.bin
set -x infersent --processes 1 --infersent-model opt/lib/infersent/infersent1.pickle  

# example invokation:
#   ungol/sentemb/training.py create $embed_glove $embed --processes 8 --redux $redux --prefix union


function __shorten -a name str
  printf "  %20s %s...\n" $name (echo "$str" | cut -b -60)
end


echo
echo "current \$dataset: $dataset"
echo "current \$redux:   $redux"
echo "current \$name:    $name"
echo
echo "exposed variables:"
__shorten '$folders' "$folders"
__shorten '$embed' "$embed"
__shorten '$embed_glove' "$embed_glove"
__shorten '$sent2vec' "$sent2vec"
__shorten '$infersent' "$infersent"
echo
echo 'additional interesting environmental things:'
echo

if set -q CUDA_VISIBLE_DEVICES
    echo "  \$CUDA_VISIBLE_DEVICES $CUDA_VISIBLE_DEVICES"
end
if set -q UNGOL_LOG
    echo "  \$UNGOL_LOG $UNGOL_LOG"
end


#
#    NOTES
#
#    create code files with embcodr in opt/codes/<DATASET>
#
# > set -x dataset sicksts
# > set -x redux mbow
# > . dotme.fish

#    current dataset: sicksts
#    current redux: mbow

#    exposed variables:
#    $folders --folder opt/data/sicksts/src/ --out opt/data/sicksts...
#    $embed --embed-provider h5py --embed-file opt/data/sicksts/mbow.emb...
#    $embed_glove --embed-provider h5py --embed-file-name opt/embed/glove-840b...

# > set src opt/current
# > set model enwiki_100k.mbow-512
# > ungol/models/embcodr.py $src/$model/compressor/model.torch $model --binary --out opt/codes/$dataset --out-prefix $model $embed
