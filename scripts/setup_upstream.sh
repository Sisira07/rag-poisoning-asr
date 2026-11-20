#!/usr/bin/env bash
# Clone the upstream TrustworthyRAG repository next to the scripts of this repo.
set -euo pipefail
cd "$(dirname "$0")/.."
if [ ! -d TrustworthyRAG ]; then
  git clone https://github.com/GPT-Laboratory/TrustworthyRAG.git TrustworthyRAG
fi
echo "Upstream ready at $(pwd)/TrustworthyRAG"
echo "The scripts find it automatically (or set TRUSTWORTHYRAG_PATH)."
