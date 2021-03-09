#!/bin/bash
#
#  V N G O L - E S
#
# elasticsearch utility script
# if something fails: delete $INSTALL_DIR
#
# usage: ./es.sh CMD [OPTS...]
#
#  where CMD is one of: install, run
#
# ---------------------------------------
#
#  CONFIG
#
#    INSTALL_DIR must be absolute
#    if run from current directory, use $(pwd)


VERSION=6.3.2
INSTALL_DIR=$(pwd)/opt/install

#
# ---------------------------------------
#

function fail {
    echo -e "\n$1\n"
    exit 2
}


# common

conf="$INSTALL_DIR/config"


# elasticsearch

es_name="elasticsearch-$VERSION"
es_file="$es_name.tar.gz"
es_url="https://artifacts.elastic.co/downloads/elasticsearch/$es_file"
es_dir="$INSTALL_DIR/es"


# kibana

kb_name="kibana-$VERSION-linux-x86_64"
kb_file="$kb_name.tar.gz"
kb_url="https://artifacts.elastic.co/downloads/kibana/$kb_file"
kb_dir="$INSTALL_DIR/kb"


#
# install <target dir> <url> <tgz file> <name>
#
function install {
    dir=$1
    url=$2
    file=$3
    name=$4

    if [ -d "$dir" ]; then
        echo "$dir already exists, skipping..."
        echo
        return
    fi

    mkdir -p "$dir"
    mkdir -p "$conf"

    echo "entering $dir"
    pushd "$dir"
    trap 'echo -e "leaving installation dir\n" && popd' EXIT

    echo -e "\nloading $url" \
      && wget "$url" \
      && echo -e "\nextracting sources\n" \
      && tar xzf "$file" \
      && echo -e "\ncopying configuration\n" \
      && cp "$name/config/"* "$conf/" \
            || fail 'did not work :('

    trap - EXIT
    popd
}


# demux

cmd="$1"

if [ "$cmd" = install ]; then
    echo; echo "installing elasticsearch"
    install $es_dir $es_url $es_file $es_name

    echo; echo "installing kibana"
    install $kb_dir $kb_url $kb_file $kb_name

elif [ "$cmd" = run ]; then

    target="$2"

    if [ "$target" = elastic ]; then
        bash -c "ES_PATH_CONF=$conf $es_dir/$es_name/bin/elasticsearch"

    elif [ "$target" = kibana ]; then
        bash -c "$kb_dir/$kb_name/bin/kibana -c $conf/kibana.yml"

    else
        fail "usage: $0 run elastic|kibana"
    fi

else
    fail "usage: $0 CMD [OPTS...]"
fi
