#!/bin/bash
# JupyterLab を 0.0.0.0:8888 で起動する。
# トークンは環境変数 JUPYTER_TOKEN で指定する。未設定なら、ランダムに生成してログに表示する。
set -e

if [ -z "${JUPYTER_TOKEN}" ]; then
  JUPYTER_TOKEN="$(python -c 'import secrets; print(secrets.token_urlsafe(24))')"
  echo "=================================================================="
  echo "JUPYTER_TOKEN が未設定のため、ランダムなトークンを生成しました:"
  echo "  ${JUPYTER_TOKEN}"
  echo "=================================================================="
fi
export JUPYTER_TOKEN

mkdir -p /workspace
cd /workspace

exec jupyter lab \
  --allow-root \
  --no-browser \
  --ip=0.0.0.0 \
  --port=8888 \
  --ServerApp.token="${JUPYTER_TOKEN}" \
  --ServerApp.allow_origin='*' \
  --ServerApp.root_dir=/workspace
