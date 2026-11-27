#!/bin/bash
# Builds the newest ik_llama.cpp (official repository) the same way as the llama.cpp reference build: Release, GGML_NATIVE=ON, CPU only.
#   bash benchmarks/night/build_ik.sh <dir>
set -eu
D=$1
mkdir -p "$D"
cd "$D"
[ -d ik_llama.cpp ] || nice -n 10 git clone --depth 1 https://github.com/ikawrakow/ik_llama.cpp ik_llama.cpp
cd ik_llama.cpp
git log -1 --format='ik_llama.cpp commit %H (%cd) %s'
nice -n 10 cmake -B build -DCMAKE_BUILD_TYPE=Release -DGGML_NATIVE=ON -DGGML_CUDA=OFF -DLLAMA_CURL=OFF
nice -n 10 cmake --build build -j 8 --target llama-server llama-bench
grep -E "^(CMAKE_BUILD_TYPE|CMAKE_CXX_COMPILER:|GGML_NATIVE|GGML_AVX|GGML_FMA|GGML_F16C|GGML_OPENMP|GGML_LTO|GGML_CPU|GGML_IQK|GGML_ARCH)" build/CMakeCache.txt || true
./build/bin/llama-server --version 2>&1 | head -5
./build/bin/llama-server --help 2>&1 | grep -iE "run-time-repack|-rtr|flash-attn|--fa|numa|--no-mmap|threads-batch|--jinja|parallel|ubatch" || true
