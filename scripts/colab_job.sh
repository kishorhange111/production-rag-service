#!/usr/bin/env bash
# Evaluate the RAG pipeline on SQuAD 2.0 on a Colab GPU runtime (A100 or L4).
#   colab new -s rag --gpu A100
#   colab upload -s rag scripts/colab_job.sh /content/colab_job.sh
#   echo 'import subprocess; subprocess.run(["bash", "/content/colab_job.sh"])' | colab exec -s rag
#
# 1. Retrieval ablation: BM25 vs dense (bge-small, bge-base) vs hybrid vs hybrid + cross-encoder reranker.
# 2. End-to-end answers with Qwen2.5-7B-Instruct, a closed-book baseline, and an LLM judge for faithfulness.
# Each step is independent; results.tgz is written on exit whatever happens.
set -uo pipefail
REPO=${REPO:-https://github.com/kishorhange111/production-rag-service}
BRANCH=${BRANCH:-main}
N=${N:-300}  # questions per type (answerable + unanswerable)
DIR=/content/production-rag-service
trap 'tar -czf /content/results.tgz -C "$DIR" results 2>/dev/null; echo "packed results ($(date -u +%T))"' EXIT
set -e
if [ ! -d "$DIR" ]; then git clone --depth 1 -b "$BRANCH" "$REPO" "$DIR"; else git -C "$DIR" fetch --depth 1 origin "$BRANCH" && git -C "$DIR" reset --hard FETCH_HEAD; fi
cd "$DIR"
pip install -q -e ".[ml]"
nvidia-smi --query-gpu=name,memory.total --format=csv
mkdir -p results
set +e

run() {  # $1 = name, rest = command; a failure is recorded and the next step still runs
  local name=$1; shift
  echo "=== $name ($(date -u +%T))"
  "$@" 2>&1 | tee "results/$name.log"
  [ "${PIPESTATUS[0]}" -eq 0 ] || echo "FAILED: $name" | tee -a results/failures.txt
}
run retrieval python -m ragservice.eval.runner retrieval --n "$N" \
  --embedders BAAI/bge-small-en-v1.5 BAAI/bge-base-en-v1.5 --reranker BAAI/bge-reranker-base --out results/retrieval.json
run answers python -m ragservice.eval.runner answers --n "$N" --embedders BAAI/bge-base-en-v1.5 \
  --generator Qwen/Qwen2.5-7B-Instruct --judge Qwen/Qwen2.5-7B-Instruct --out results/answers.json
echo "DONE $(date -u +%T)"; cat results/failures.txt 2>/dev/null
