#!/usr/bin/env bash
# Download papers that are cited in references.bib but not stored in this repo
# (their licenses don't allow redistribution). Files go to literature/local/,
# which is git-ignored.
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p local

arxiv=(
  "1910.11006 li2020_wlasl"
  "2308.12419 shi2023_asl-real-world-thesis"
  "2407.01032 traub2024_selective-classification"
  "1908.08597 bragg2019_interdisciplinary-perspective"
  "1906.08172 lugaresi2019_mediapipe"
  "2207.04672 nllb2022"
  "2212.04356 radford2022_whisper"
)

for entry in "${arxiv[@]}"; do
  read -r id name <<<"$entry"
  out="local/${name}.pdf"
  [[ -f "$out" ]] && { echo "skip  $out"; continue; }
  echo "fetch $out"
  curl -sSfL -o "$out" "https://arxiv.org/pdf/${id}"
  sleep 3  # be polite to arXiv
done

out="local/zhang_gloss-translation-low-resource-mt.pdf"
[[ -f "$out" ]] || curl -sSfL -o "$out" "https://www.cs.jhu.edu/~xzhan138/papers/SLMT_Book_G2T.pdf"

echo "Done. Paywalled papers (see README) must be downloaded manually into literature/local/."
