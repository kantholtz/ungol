# source this


function run_retrieval -a no

    if [ "$no" -lt 1 -o 3 -lt "$no" ]
        echo 'provide dataset no 1, 2 or 3'
        exit 2
    end

    set -l f_truth "data/CLEF/truth/CLEF2003_ah-mono-de.txt"
    set -l f_out "opt/current/$no"
    set -l f_index "opt/indexes/18.11.28_fasttext.de-256.index.pickle"
    set -l argv "$f_truth" "$f_out" --ungol --ungol-index "$f_index" -k 250

    mkdir -p "$f_out"

    echo
    for dataset in opt/datasets/180921.*.$no.pickle
        echo $dataset
        ./ungol/retrieval/evaluate.py $argv --dataset $dataset | tee "$f_out"/(basename "$dataset").log
        echo
    end

end
