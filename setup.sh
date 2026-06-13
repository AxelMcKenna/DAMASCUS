#!/usr/bin/env bash
# Fetches third-party dependencies: metal-cpp headers and llama.cpp baseline.
set -euo pipefail

THIRD_PARTY=third_party
mkdir -p "$THIRD_PARTY"

# ---- metal-cpp (Apple-official C++ bindings, header-only) ----
if [ ! -d "$THIRD_PARTY/metal-cpp" ]; then
    echo "==> fetching metal-cpp..."
    pushd "$THIRD_PARTY" > /dev/null
    # If this URL ages out, grab the latest from https://developer.apple.com/metal/cpp/
    curl -L -o metal-cpp.zip \
        "https://developer.apple.com/metal/cpp/files/metal-cpp_macOS15_iOS18.zip"
    unzip -q metal-cpp.zip
    rm metal-cpp.zip
    popd > /dev/null
    echo "    metal-cpp ready at $THIRD_PARTY/metal-cpp"
else
    echo "==> metal-cpp already present, skipping"
fi

# ---- llama.cpp (baseline to beat) ----
if [ ! -d "$THIRD_PARTY/llama.cpp" ]; then
    echo "==> cloning + building llama.cpp..."
    pushd "$THIRD_PARTY" > /dev/null
    git clone --depth 1 https://github.com/ggerganov/llama.cpp.git
    pushd llama.cpp > /dev/null
    # llama.cpp's build system; Metal is on by default on Apple Silicon now
    cmake -B build -DGGML_METAL=ON -DCMAKE_BUILD_TYPE=Release
    cmake --build build --config Release -j 8
    popd > /dev/null
    popd > /dev/null
    echo "    llama.cpp built at $THIRD_PARTY/llama.cpp/build"
else
    echo "==> llama.cpp already present, skipping"
fi

cat <<'EOF'

Done. Next steps:

  1) Download Llama-3.2-3B-Instruct GGUF (Q4_K_M):
     mkdir -p models
     curl -L -o models/llama-3.2-3b-instruct.Q4_K_M.gguf \
       https://huggingface.co/bartowski/Llama-3.2-3B-Instruct-GGUF/resolve/main/Llama-3.2-3B-Instruct-Q4_K_M.gguf

  2) Run baseline benchmarks (see benchmarks/baseline.md for methodology):
     ./third_party/llama.cpp/build/bin/llama-bench \
       -m models/llama-3.2-3b-instruct.Q4_K_M.gguf \
       -p 1024 -n 256 -r 5

  3) Record numbers in benchmarks/baseline.md. These are your forever-baselines.

EOF
