#!/bin/bash

# base=..
# exp=opt/experiments/tau

# the following arguments must be provided:
#
#  -- base: /path/to/ungol-models git directory
#  -- conf: name of the config file

base="$1"
conf="$2"

if [ -z "$base" -o -z "$conf" ]; then
    echo "usage: $0 BASE CONF"
    exit 2
fi

name=$(basename $conf .conf)
exp="opt/experiments/$name"

echo "running experiment series for '$name'"
echo "saving stats to '$base/$exp'"

echo 'clearing logfile'
echo > embcompr.log

echo 'removing former stats'
rm -r $base/$exp 2>/dev/null

echo 'running experiments'
python $base/ungol/embcompr.py "$conf"

echo 'exiting run.sh'
