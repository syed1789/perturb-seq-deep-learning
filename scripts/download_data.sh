#!/bin/bash
# Fetches one sample of the Adamson et al. 2016 Perturb-seq dataset
# (harmonized to h5ad by the scPerturb project) from Zenodo.
set -euo pipefail
cd "$(dirname "$0")/.."

URL="https://zenodo.org/api/records/13350497/files/AdamsonWeissman2016_GSM2406675_10X001.h5ad/content"
OUT="data/adamson2016_10X001.h5ad"
EXPECTED_MD5="232f7e3756d41602bbe434b50662a76f"

mkdir -p data
if [ -f "$OUT" ]; then
  echo "Already downloaded: $OUT"
else
  echo "Downloading $OUT (~35MB)"
  curl -sL --fail -o "$OUT" "$URL"
fi

actual_md5="$(md5sum "$OUT" 2>/dev/null | awk '{print $1}' || md5 -q "$OUT")"
if [ "$actual_md5" != "$EXPECTED_MD5" ]; then
  echo "MD5 mismatch: expected $EXPECTED_MD5, got $actual_md5" >&2
  exit 1
fi
echo "MD5 verified: $actual_md5"
