#!/usr/bin/env bash

PWD=$(pwd)
TT_CACHE_DIR=$(pwd)/tt_cache
export TRITON_CACHE_DIR=$PWD/tt_cache
export MLIR_ENABLE_DUMP=1
export INPATH=$PWD/scripts/triton_reproducer/

if [ -d "$TT_CACHE_DIR" ]; then
    rm -rf "$TT_CACHE_DIR"
fi

TTREPEXE_DIR=$PWD/scripts/triton_reproducer/
rm $TTREPEXE_DIR/tritonspvc++.exe
rm -rf $TTREPEXE_DIR/data
mkdir $TTREPEXE_DIR/data

# Run user provided pytest
$1

SEARCH_DIR=$TT_CACHE_DIR
temp_file=$(mktemp)

find "$SEARCH_DIR" -type f -name "*.spv" > "$temp_file"

while IFS= read -r file_path; do
    echo "Processing file: $file_path"
    cp $file_path $PWD/scripts/triton_reproducer/data/kernel.spv
    pushd $PWD/scripts/triton_reproducer > /dev/null
    make
    popd > /dev/null
    $TTREPEXE_DIR/tritonspvc++.exe -spv $TTREPEXE_DIR/data/kernel.spv
done < "$temp_file"
rm "$temp_file"
