# syntax=docker/dockerfile:1.4
# =====================================================================
# 部屋の写真 → 3Dシーン再構成パイプライン用イメージ(RunPod向け)
#
# 入れるもの : torch 2.5.1 / pipパッケージ一式 / ビルド済み vis4d_cuda_ops(CUDA版)/ WildDet3D / JupyterLab
# 入れないもの: モデルの重み(SAM3・WildDet3D等)、Hugging Faceトークン、入力・出力ファイル
#              → 重みは /workspace(Persistent storage)に保存し、トークンは実行時に入力する
# =====================================================================
FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    CUDA_HOME=/usr/local/cuda \
    TORCH_CUDA_ARCH_LIST="7.5;8.0;8.6;8.9;9.0+PTX" \
    MAX_JOBS=4 \
    WILDDET3D_DIR=/opt/WildDet3D \
    IN_DOCKER_IMAGE=1

RUN apt-get update \
 && apt-get install -y --no-install-recommends git unzip curl ca-certificates libgl1 libglib2.0-0 \
 && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip wheel ninja "setuptools<80"

# ---- ノートブック Section 1 と同じ順序でインストール(依存関係の解決結果をそろえるため) ----
RUN pip install huggingface_hub objaverse open_clip_torch
RUN pip install trimesh pyfqmr matplotlib scipy networkx shapely
RUN pip install "transformers==5.15.1" accelerate timm
RUN pip install -U safetensors

# ---- Section 1.5: MoGe-2 ----
RUN pip install opencv-python-headless
RUN pip install "git+https://github.com/microsoft/MoGe.git"

# ---- Section 7.5: vis4d / vis4d_cuda_ops / WildDet3D ----
RUN pip install vis4d==1.0.0

# vis4d_cuda_ops の setup.py は「ビルド時にGPUが見えないとCPU専用版を作る」ため、
# Dockerのビルド環境(GPUなし)でもCUDA版を作れるよう、その判定だけを外す。
# 対象GPU世代は TORCH_CUDA_ARCH_LIST で明示している。
RUN git clone --depth 1 https://github.com/SysCV/vis4d_cuda_ops.git /tmp/vis4d_cuda_ops \
 && cd /tmp/vis4d_cuda_ops \
 && sed -i 's/if torch.cuda.is_available() and CUDA_HOME is not None:/if CUDA_HOME is not None:/' setup.py \
 && grep -q 'if CUDA_HOME is not None:' setup.py \
 && pip install . --no-build-isolation \
 && cd / && rm -rf /tmp/vis4d_cuda_ops

# CUDA版としてビルドされているか検証する(CPU専用版ならここでビルドを失敗させる)
RUN python - <<'PYEOF'
import re, subprocess, sys
import vis4d_cuda_ops
so = vis4d_cuda_ops.__file__
out = subprocess.run(["/usr/local/cuda/bin/cuobjdump", "--list-elf", so], capture_output=True, text=True).stdout
archs = sorted(set(re.findall(r"sm_(\d+)", out)))
print("vis4d_cuda_ops:", so)
print("埋め込まれたSASSの世代:", archs)
if not archs:
    sys.exit("vis4d_cuda_ops がCUDA版になっていません(CPU専用版)。")
PYEOF

RUN git clone --recurse-submodules https://github.com/allenai/WildDet3D.git ${WILDDET3D_DIR}
RUN pip install -r ${WILDDET3D_DIR}/requirements.txt

# ---- 以降のセクションで使うパッケージ(plotly / CoACD / rtree / python-fcl)と、JupyterLab ----
RUN pip install plotly coacd rtree python-fcl
RUN pip install jupyterlab ipywidgets

# ---- 最終確認: torch が 2.5.1 のままか(他パッケージに入れ替えられていないか) ----
RUN python - <<'PYEOF'
import torch, transformers, trimesh, scipy, fcl, rtree, coacd, open_clip, objaverse, moge, vis4d_cuda_ops
print("torch", torch.__version__, "| CUDA", torch.version.cuda, "| transformers", transformers.__version__)
assert torch.__version__.startswith("2.5.1"), f"torch が入れ替わっています: {torch.__version__}"
PYEOF
RUN pip check || true

COPY start.sh /usr/local/bin/start.sh
RUN sed -i 's/\r$//' /usr/local/bin/start.sh \
 && chmod +x /usr/local/bin/start.sh \
 && mkdir -p /workspace

WORKDIR /workspace
EXPOSE 8888
CMD ["/usr/local/bin/start.sh"]
