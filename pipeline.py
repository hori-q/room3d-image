"""
部屋の写真から、壁・家具(可視化用+物理コライダー用)のGLBを生成するパイプライン。
RunPod Serverless向け。handler.py から import して使う。

構成:
- load_models(): コールドスタート時に1回だけ呼ぶ。モデル・HSSD家具DBをすべてGPU上に
  読み込んで、モジュールのグローバル変数として保持する(ウォームなワーカーで使い回す)。
- run_pipeline(image_bytes, options, progress): リクエストごとに呼ぶ。1枚の画像を
  受け取り、壁・可視化用メッシュ・コライダー用メッシュ・マニフェストを生成して返す。

未検証: このファイルは、元のJupyterノートブックの各セルをコード順に連結して
作成したもので、GPU環境での実行確認はできていません。特に、"del" していたモデルを
すべて常駐させる構成へ変えた点は、元のノートブックでは想定されていなかった変更です。
初回はローカルテスト(このファイル末尾のtest_input.jsonでの単体実行)を必ず行ってください。
"""
import base64
import gc
import json
import os
import shutil
import tempfile
import time
import uuid


# 進捗の割合(目安。実測してから調整してください)
_PCT = {
    "depth": 0,
    "detect_sam3": 15,
    "wilddet3d": 35,
    "clip_map": 55,
    "placement": 70,
    "coacd": 80,
    "unity_export": 95,
}


def _noop_progress(_payload):
    pass


# ============================================================
# 両関数(load_models / run_pipeline)から共通して使うimportは、ここに集約する。
# (関数の内側でimportすると、その関数のローカル変数として扱われ、別の関数から
#  読めなくなる。とくに finally節にまたがる名前は、関数内で一度でも import される
#  だけで UnboundLocalError の原因になるため、ここでまとめて済ませておく。)
# ============================================================
import gzip
import io
import sys
from collections import Counter

import cv2
import numpy as np
import torch
import trimesh
import trimesh.collision
import trimesh.transformations as _tf
import pyfqmr
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

import objaverse
import open_clip
import huggingface_hub
from huggingface_hub import hf_hub_download
import transformers
import safetensors
from transformers import AutoProcessor
from transformers import OmDetTurboForObjectDetection
from transformers import Sam3TrackerProcessor, Sam3TrackerModel

from moge.model.v2 import MoGeModel  # MoGe-2

# WildDet3D(専用イメージの/opt/WildDet3Dに、サブモジュール込みで導入済み)。
# sys.pathへの追加は、load_models()の実行有無に関係なく、このモジュールが
# importされた時点(＝コンテナ起動時)で一度だけ行う。
WILDDET3D_DIR = os.environ.get("WILDDET3D_DIR", "/opt/WildDet3D")
if os.path.isdir(WILDDET3D_DIR) and WILDDET3D_DIR not in sys.path:
    sys.path.insert(0, WILDDET3D_DIR)

import vis4d_cuda_ops
from wilddet3d import build_model, preprocess
from wilddet3d.data_types import WildDet3DInput
from wilddet3d.inference import _orig_to_input_hw_box, _pairwise_iou


def load_models():
    """コールドスタート時に1回だけ呼ぶ。以降のrun_pipeline()呼び出しは、ここで読み込んだモデルを使い回す。"""
    global BASE_DIR, INPUT_DIR, device, USE_BF16_AUTOCAST, bf16_autocast, HF_TOKEN, moge_model, MOGE_MODEL_ID, omdet_processor, omdet_model, sam3_tracker_model, sam3_tracker_processor, DETECTOR_MODEL_ID, wilddet3d_model, WILDDET3D_SCORE_THRESHOLD, WILDDET3D_SCORE_3D_THRESHOLD, OBB_DEPTH_CORRECTION_ENABLED, OBB_DEPTH_CORRECTION_MIN_POINTS, OBB_DEPTH_CORRECTION_PERCENTILE, OBB_DEPTH_CORRECTION_MAX_IQR_MULT, OBB_DEPTH_CORRECTION_GAIN, clip_model, clip_tokenizer, clip_preprocess, CLIP_ARCH, CLIP_PRETRAINED, OPEN_CLIP_CACHE_DIR, db_embeddings, db_names, db_relpaths, db_categories, db_bbox_whd, WILDDET3D_DIR, WILDDET3D_CKPT_PATH, WILDDET3D_CKPT_FILENAME, WILDDET3D_CKPT_DIR, CACHE_ROOT, MODEL_CACHE_DIR

    # ==================== 0. 環境確認・作業ディレクトリ ====================

    import os, subprocess, sys

    # ---- 1) 専用イメージで起動しているか ----
    if os.environ.get("IN_DOCKER_IMAGE") != "1":
        raise RuntimeError("専用Dockerイメージ用のpipeline.pyです(環境変数 IN_DOCKER_IMAGE がありません)。")

    # ---- 2) GPU世代の確認 ----
    try:
        _smi = subprocess.run(["nvidia-smi", "--query-gpu=name,compute_cap", "--format=csv,noheader"],
                              capture_output=True, text=True, timeout=30).stdout.strip()
    except Exception:
        _smi = ""
    print("GPU:", _smi or "(nvidia-smiで取得できませんでした)")
    for _line in _smi.splitlines():
        try:
            _cap = float(_line.split(",")[-1])
        except ValueError:
            continue
        if _cap >= 10.0:
            raise RuntimeError(
                f"このGPU({_line.strip()})はBlackwell世代です。イメージに入っている torch 2.5.1 は Blackwell に非対応のため、"
                "このまま進めても動きません。Ampere/Ada/Hopper世代のGPU(RTX 3090・4090、A5000、A40、L4、A100、H100など)で"
                "Podを作り直してください。")

    # ---- 3) バージョンの表示 ----
    import importlib.metadata as _m
    print("torch:", _m.version("torch"))
    print("vis4d_cuda_ops の対応GPU世代:", os.environ.get("TORCH_CUDA_ARCH_LIST"))
    print("WildDet3D:", os.environ.get("WILDDET3D_DIR"))

    # ---- 4) bf16(半精度)推論の切り替え(有効・無効の指定だけ。torch はまだimportしていない) ----
    # Ampere世代以降(RTX 3090/4090・A5000・A40・L4・A100・H100など)は、bf16での推論に対応している。
    # KaggleのT4は非対応だったため、これまで使えなかった最適化。
    # USE_BF16_AUTOCAST を False にすれば、いつでも今までどおり(fp32)の動作に戻せる。
    # 実際に有効化する処理(bf16_autocast の定義など)は、torch をimportした直後のセルで行う。
    USE_BF16_AUTOCAST = True
    print("USE_BF16_AUTOCAST:", USE_BF16_AUTOCAST)


    import os

    # GPUメモリの断片化を軽減する設定(OOM対策の補助。根本対策はモデルを同時に載せすぎないこと)
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("Device:", torch.cuda.get_device_name(0))
    print("torch:", torch.__version__, "cuda:", torch.version.cuda)

    # Serverlessでは、Network Volumeは /runpod-volume にマウントされる。
    # モデルの重み・HSSD家具DBは、ここに永続化しておく(ワーカーが再利用されるたびに使い回す)。
    # /workspace は、通常のPod(このpipeline.pyを動作確認するためのテスト環境)向けのフォールバック。
    if os.path.isdir("/runpod-volume"):
        BASE_DIR = "/runpod-volume"
    elif os.path.isdir("/workspace"):
        BASE_DIR = "/workspace"
    else:
        BASE_DIR = "/tmp"  # ローカルテスト用のフォールバック(重みは毎回ダウンロードし直しになる)

    # 入力ファイル(部屋の写真・HSSD家具DB)を置くフォルダ。JupyterLabのファイルパネルから
    # ここへアップロードしてください。
    INPUT_DIR = os.path.join(BASE_DIR, "inputs")
    os.makedirs(INPUT_DIR, exist_ok=True)
    print("input dir  :", INPUT_DIR)

    # WORK_DIR はリクエストごとに作るため、ここでは作らない(run_pipeline()内で決める)。


    # ==================== bf16 autocast ヘルパー ====================

    import contextlib


    def bf16_autocast():
        """USE_BF16_AUTOCAST が True のときだけ、torch.autocast(bf16) を有効にする。"""
        if USE_BF16_AUTOCAST:
            return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        return contextlib.nullcontext()


    # Ampere世代以降では、float32の行列演算も少し速くなる(精度はわずかに下がる)
    if USE_BF16_AUTOCAST:
        torch.set_float32_matmul_precision("high")


    # ==================== モデルキャッシュ設定 ====================

    import shutil

    # ==========================================
    # RunPodでは通常 None のままでOK(/workspace/model_cache_export に自動で溜まり、次回も使われる)。
    # 別の場所に用意済みのキャッシュ(model_cache/ と WildDet3D/ が入ったフォルダ)を
    # コピーして使いたい場合だけ、そのパスを指定する。
    CACHE_INPUT_DIR = None
    # ==========================================

    # 実行結果(room_furniture_pipeline以下の検出画像・検索結果・シーンglbなど)と混ざらないよう、
    # キャッシュ対象だけを1つの専用フォルダ(model_cache_export/)にまとめる。
    # キャッシュ(再取得できるもの)と実行結果(room_furniture_pipeline/)を分けて管理するためのフォルダ。
    CACHE_ROOT = os.path.join(BASE_DIR, "model_cache_export")
    os.makedirs(CACHE_ROOT, exist_ok=True)

    MODEL_CACHE_DIR = os.path.join(CACHE_ROOT, "model_cache")
    os.makedirs(MODEL_CACHE_DIR, exist_ok=True)

    _cached_model_src = os.path.join(CACHE_INPUT_DIR, "model_cache") if CACHE_INPUT_DIR else None
    if _cached_model_src and os.path.isdir(_cached_model_src):
        print(f"[cache] {_cached_model_src} からモデルキャッシュをコピーします(初回のみ)...")
        for name in os.listdir(_cached_model_src):
            src = os.path.join(_cached_model_src, name)
            dst = os.path.join(MODEL_CACHE_DIR, name)
            if os.path.exists(dst):
                continue
            if os.path.isdir(src):
                shutil.copytree(src, dst)
            else:
                shutil.copy2(src, dst)
        print("[cache] モデルキャッシュのコピーが完了しました:", MODEL_CACHE_DIR)
    else:
        print("[cache] 入力キャッシュが指定されていないため、初回ダウンロードを行います。")
        print("[cache] 今回のダウンロード結果は", MODEL_CACHE_DIR, "に保存されます。")
        print("[cache] このフォルダはNetwork Volume上にあるため、ワーカーが再利用される限り残ります。")

    # transformers / huggingface_hub / MoGe(内部でhf_hub_downloadを使用)は、
    # 以下の環境変数を見てキャッシュの読み書き先を決める。
    # これを差し替えるだけで、Section 3(OmDet-Turbo・SAM3 Tracker)やSection 1.5(MoGe-2)の
    # from_pretrained呼び出しは一切変更せずにキャッシュが効くようになる。
    _hf_cache_dir = os.path.join(MODEL_CACHE_DIR, "huggingface")
    os.environ["HF_HOME"] = _hf_cache_dir
    os.environ["HF_HUB_CACHE"] = os.path.join(_hf_cache_dir, "hub")
    os.environ["TRANSFORMERS_CACHE"] = os.path.join(_hf_cache_dir, "hub")

    # open_clip(Section 10.1のCLIP)用のキャッシュ先。こちらは呼び出し側で
    # cache_dir=OPEN_CLIP_CACHE_DIR を明示的に渡す必要がある(Section 10.1側で対応済み)。
    OPEN_CLIP_CACHE_DIR = os.path.join(MODEL_CACHE_DIR, "open_clip")
    os.makedirs(OPEN_CLIP_CACHE_DIR, exist_ok=True)

    print("HF_HOME:", os.environ["HF_HOME"])
    print("OPEN_CLIP_CACHE_DIR:", OPEN_CLIP_CACHE_DIR)


    # ==================== 依存パッケージの確認・import ====================

    import sys, subprocess, importlib, importlib.util

    # RunPodのPyTorchテンプレートには scipy / networkx / shapely が入っていない(Kaggleには入っていた)。
    # trimesh は import 時に scipy が無いと「使えない」と記憶してしまい、後からインストールしても
    # 直らないため、trimesh を import する前に、足りなければここで入れる。
    _missing = [m for m in ("scipy", "networkx", "shapely") if importlib.util.find_spec(m) is None]
    if _missing:
        if "trimesh" in sys.modules:
            raise RuntimeError(
                f"{_missing} が未インストールで、trimesh は既にimport済みです。"
                "「Kernel → Restart Kernel」でカーネルを再起動し、このセルから実行し直してください。"
            )
        print("不足パッケージをインストールします:", _missing)
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *_missing])
        importlib.invalidate_caches()

    print("objaverse:", objaverse.__version__)
    print("transformers:", transformers.__version__)
    print("huggingface_hub:", huggingface_hub.__version__)
    print("safetensors:", safetensors.__version__)
    print("OmDetTurboForObjectDetection / Sam3TrackerModel のimport確認: OK")


    # ==================== Hugging Face 認証 ====================

    # --- Hugging Face認証: Serverlessには入力欄が無いため、環境変数(Endpointの
    #     Secret)からのみ読む。対話的な入力にはフォールバックしない。 ---
    HF_TOKEN = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not HF_TOKEN:
        raise RuntimeError(
            "HF_TOKEN が環境変数に設定されていません。RunPodのServerless Endpoint作成時に、"
            "Environment Variables(Secret推奨)として HF_TOKEN を設定してください。"
            "(facebook/sam3 と hssd/hssd-hab は、事前にHugging Face上でアクセス申請の承認が必要です。)"
        )
    os.environ["HF_TOKEN"] = HF_TOKEN
    os.environ["HUGGING_FACE_HUB_TOKEN"] = HF_TOKEN
    print("Hugging Face token を環境変数として設定しました。")


    # ==================== MoGe-2(深度推定)ロード ====================

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    MOGE_MODEL_ID = "Ruicheng/moge-2-vitl-normal"  # MoGe-2 (ViT-L、法線推定つき、メートルスケール)

    moge_model = MoGeModel.from_pretrained(MOGE_MODEL_ID).to(device).eval()
    print("MoGe-2 loaded:", MOGE_MODEL_ID, "on", device)


    # ==================== OmDet-Turbo + SAM3 Tracker ロード ====================

    import time
    _proc_t0 = time.time()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- OmDet-Turbo(検出用。transformers本体に統合されている唯一の公式チェックポイント) ---
    DETECTOR_MODEL_ID = "omlab/omdet-turbo-swin-tiny-hf"
    omdet_processor = AutoProcessor.from_pretrained(DETECTOR_MODEL_ID)
    omdet_model = OmDetTurboForObjectDetection.from_pretrained(DETECTOR_MODEL_ID).to(device).eval()

    # --- SAM3 Tracker(ボックス -> マスク用) ---
    sam3_tracker_model = Sam3TrackerModel.from_pretrained("facebook/sam3").to(device).eval()
    sam3_tracker_processor = Sam3TrackerProcessor.from_pretrained("facebook/sam3")

    print("Detector:", DETECTOR_MODEL_ID, "loaded on:", device)
    print("SAM3 Tracker loaded on:", device)


    print(f"\n\u23f1\ufe0f [Section 3 モデルロード(OmDet-Turbo + SAM3 Tracker)] \u51e6\u7406\u6642\u9593: {time.time() - _proc_t0:.2f}\u79d2")


    # ==================== WildDet3D ロード ====================
    # --- WildDet3D 本体は、専用イメージの /opt/WildDet3D に(サブモジュール込みで)導入済み ---
    WILDDET3D_DIR = os.environ.get("WILDDET3D_DIR", "/opt/WildDet3D")
    assert os.path.isdir(WILDDET3D_DIR), f"{WILDDET3D_DIR} が見つかりません。専用イメージで起動していますか?"
    print("WildDet3D:", WILDDET3D_DIR)

    # --- 依存関係(torch 2.5.1・vis4d・vis4d_cuda_ops・requirements.txt)も、イメージに導入済み ---
    _cap = torch.cuda.get_device_capability()
    _archs = os.environ.get("TORCH_CUDA_ARCH_LIST", "").replace("+PTX", "").split(";")
    if f"{_cap[0]}.{_cap[1]}" not in _archs:
        print(f"[警告] このGPU(sm_{_cap[0]}{_cap[1]})向けのコードが vis4d_cuda_ops に含まれていない可能性があります"
              f"(含まれる世代: {_archs})。実行時に「no kernel image」というエラーが出たら、イメージのビルドし直しが必要です。")
    print("vis4d_cuda_ops: OK")

    # --- 学習済みチェックポイント(~4.7GB)のダウンロード ---

    WILDDET3D_CKPT_FILENAME = "wilddet3d_alldata_all_prompt_v1.0.pt"
    # イメージ内(/opt)は、Stopで消えるため、チェックポイントは /workspace 側に置く
    WILDDET3D_CKPT_DIR = os.path.join(CACHE_ROOT, "wilddet3d_ckpt")
    WILDDET3D_CKPT_PATH = os.path.join(WILDDET3D_CKPT_DIR, WILDDET3D_CKPT_FILENAME)

    if os.path.exists(WILDDET3D_CKPT_PATH):
        print("[cache] チェックポイントは既に存在します(ダウンロード省略):", WILDDET3D_CKPT_PATH)
    else:
        WILDDET3D_CKPT_PATH = hf_hub_download(
            repo_id="allenai/WildDet3D",
            filename=WILDDET3D_CKPT_FILENAME,
            local_dir=WILDDET3D_CKPT_DIR,
        )

    print("checkpoint:", WILDDET3D_CKPT_PATH)



    # ==========================================
    WILDDET3D_SCORE_THRESHOLD = 0.3      # 2Dスコアのしきい値(geometricプロンプトでは実質未使用)
    WILDDET3D_SCORE_3D_THRESHOLD = 0.1   # 3Dスコアのしきい値(同上)

    # --- OBBの奥行き(カメラからの距離)を、MoGe-2の深度点群を使って補正する設定 ---
    # WildDet3Dは画像上の位置(方向)は比較的正確だが、単眼推定特有の奥行きのズレが
    # 出やすい。そこで、方向は変えずに「カメラ中心からOBB中心へのベクトル」の長さ
    # だけを、実際に観測された深度点群に合わせて調整する。
    OBB_DEPTH_CORRECTION_ENABLED = True
    OBB_DEPTH_CORRECTION_MIN_POINTS = 30      # マスク内でこれ未満しか有効な深度点が無ければ補正しない
    OBB_DEPTH_CORRECTION_PERCENTILE = 25.0    # 外れ値除去後、この分位点(低め)を「手前面」の代表距離とする
    OBB_DEPTH_CORRECTION_MAX_IQR_MULT = 2.5   # 外れ値除去のしきい値(四分位範囲の何倍まで許容するか)
    OBB_DEPTH_CORRECTION_GAIN = 0.8           # 補正量に掛けるゲイン(1.0で完全に合わせる、小さくすると控えめに)
    # ==========================================

    # 複数ボックスをバッチ化すると、NMSが同じ画像内の別の家具の候補まで巻き込んで誤って
    # 抑制してしまう可能性がある(1件ずつ呼んでいた従来実装では起きなかった問題)。
    # geometricモードではどのみちボックスごとにIoUマッチングで個別に候補を選び直すため、
    # NMSはオフにしてこのリスクを避ける。
    wilddet3d_model = build_model(
        checkpoint=WILDDET3D_CKPT_PATH,
        score_threshold=WILDDET3D_SCORE_THRESHOLD,
        score_3d_threshold=WILDDET3D_SCORE_3D_THRESHOLD,
        nms=False,                    # 複数ボックスバッチ時のNMS巻き込みを避ける
        skip_pretrained=True,         # チェックポイントにSAM3/LingBot-Depthの重みが既に含まれているため
        use_depth_input_test=True,    # MoGe-2の深度マップをdepth_gtとして使うため
    )
    print("WildDet3D loaded from:", WILDDET3D_CKPT_PATH)


    # ==================== CLIP ロード ====================

    import time
    _proc_t0 = time.time()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # HSSD版DB(furniture_db_hssd.npz)は hssd_db_build.ipynb で ViT-L-14/openai を使って
    # 画像埋め込みを計算しているため、ここも必ず同じモデルを使う(埋め込み次元・空間が
    # 一致していないと、コサイン類似度に意味がなくなる。次元数も768で異なるため一致していないと
    # そもそも内積計算でエラーになる)。
    # (以前のOpenShape/objaverse.pt版はViT-bigG-14とペアだったが、HSSD版には適用されない)
    CLIP_ARCH, CLIP_PRETRAINED = "ViT-L-14", "openai"

    # clip_preprocess: Section 11.3でクロップ画像をCLIPの画像エンコーダに通す際の前処理(preprocess_val)
    clip_model, _, clip_preprocess = open_clip.create_model_and_transforms(
        CLIP_ARCH, pretrained=CLIP_PRETRAINED, cache_dir=OPEN_CLIP_CACHE_DIR,
    )
    clip_model = clip_model.to(device).eval()
    clip_tokenizer = open_clip.get_tokenizer(CLIP_ARCH)

    print(f"loaded CLIP: {CLIP_ARCH} ({CLIP_PRETRAINED})  (テキストエンコーダ + 画像エンコーダの両方を使用)")


    print(f"\n\u23f1\ufe0f [Section 10.1 CLIPモデルロード] \u51e6\u7406\u6642\u9593: {time.time() - _proc_t0:.2f}\u79d2")


    # ==================== HSSD家具DB 読み込み ====================

    import time
    _proc_t0 = time.time()

    # ==========================================
    # hssd_db_build.ipynb で構築した furniture_db_hssd.npz の場所を指定する。
    # RunPodでは furniture_db_hssd_size.npz を /workspace/inputs/ (= INPUT_DIR) にアップロードしておく。
    # Noneのままなら WORK_DIR と INPUT_DIR を自動探索する。別の場所に置いた場合だけそのフォルダを指定する。
    HSSD_DB_INPUT_DIR = None  # 例: "/workspace/my_db_folder"
    # ==========================================


    def _find_hssd_db(explicit_dir=None, search_root=INPUT_DIR, filename="furniture_db_hssd_size.npz"):
        if explicit_dir:
            candidate = os.path.join(explicit_dir, filename)
            if os.path.exists(candidate):
                return candidate
        # (load_models()時点ではリクエスト単位のWORK_DIRがまだ存在しないため、
        #  ここではsearch_root(INPUT_DIR)だけを探索する)
        if os.path.isdir(search_root):
            for root, _dirs, files in os.walk(search_root):
                if filename in files:
                    return os.path.join(root, filename)
        return None


    _hssd_db_path = _find_hssd_db(HSSD_DB_INPUT_DIR)
    if _hssd_db_path is None:
        raise FileNotFoundError(
            "furniture_db_hssd.npz が見つかりません。hssd_db_build.ipynb で事前に構築し、"
            f"そのファイルを {INPUT_DIR} にアップロードするか、"
            "HSSD_DB_INPUT_DIR にフォルダのパスを指定してください。"
        )

    _hssd_db = np.load(_hssd_db_path, allow_pickle=True)
    db_embeddings = _hssd_db["embeddings"].astype(np.float32)
    db_names = _hssd_db["names"]           # HSSDのオブジェクトUID(ファイル名のハッシュ部分)
    db_relpaths = _hssd_db["relpaths"]     # HSSDリポジトリ内の相対パス(GLB取得に使う。Section 12で使用)
    db_categories = None                    # HSSD版DBにはカテゴリラベルが無いため絞り込みは行わない

    # バウンディングボックス寸法(dx, dy, dz)[m]。サイズ絞り込み検索(Section 10.3)で使う。
    # 古いバージョンのDB(hssd_db_build.ipynbのSection 8.5より前に作ったもの)には
    # 含まれていないことがあるため、無ければNoneにしてサイズ絞り込みを自動的に無効化する。
    db_bbox_whd = _hssd_db["bbox_whd"].astype(np.float32) if "bbox_whd" in _hssd_db.files else None

    print(f"\u2705 HSSD家具DBを読み込みました: {_hssd_db_path}")
    print(f"DB embeddings shape: {db_embeddings.shape}")
    print(f"DB件数: {len(db_names)}")
    if db_bbox_whd is not None:
        _n_valid_bbox = int(np.sum(~np.isnan(db_bbox_whd).any(axis=1)))
        print(f"バウンディングボックス寸法あり: {_n_valid_bbox} / {len(db_names)}件"
              "(サイズ絞り込み検索が使えます)")
    else:
        print("このDBにはバウンディングボックス寸法が含まれていません(サイズ絞り込み検索は無効になります)。")


    print(f"\n\u23f1\ufe0f [Section 4 HSSD家具DB読み込み] \u51e6\u7406\u6642\u9593: {time.time() - _proc_t0:.2f}\u79d2")


    print("=== load_models() 完了 ===")


def run_pipeline(image_bytes, options=None, progress=None):
    """
    image_bytes: 部屋の写真(PNG/JPEGのバイト列)
    options: {
        "hssd_db_path": HSSD家具DBのパス(省略時はload_models()で読み込んだものを使う。今は未使用),
        "coacd_fast_mode": bool(既定 True),
    }
    progress: dict を受け取るコールバック(RunPodの progress_update に渡す用)。省略可。
    戻り値: {
        "walls_glb_base64": ..., "visual_glb_base64": ..., "collision_glb_base64": ...,
        "manifest": {...},  # unity_manifest.json の中身
        "job_work_dir": WORK_DIR (デバッグ用。呼び出し側では通常不要)
    }
    """
    options = options or {}
    progress = progress or _noop_progress

    def report(payload):
        progress(payload)

    COACD_FAST_MODE = bool(options.get("coacd_fast_mode", True))

    # リクエストごとに、独立した作業ディレクトリを使う(並行実行時に他のジョブと混ざらないように)
    WORK_DIR = os.path.join(BASE_DIR, "jobs", uuid.uuid4().hex)
    os.makedirs(WORK_DIR, exist_ok=True)
    os.chdir(WORK_DIR)

    image_path = os.path.join(WORK_DIR, "input_image.png")
    with open(image_path, "wb") as f:
        f.write(image_bytes)

    try:
        progress({"percent": _PCT["depth"], "label": "深度推定中"})

        # ---- depth (元ノートブック cell 15) ----
        # 画像を読み込み、MoGe-2の入力形式(RGB, [0,1]正規化, (3,H,W)のtensor)に変換
        _bgr = cv2.imread(image_path)
        if _bgr is None:
            raise FileNotFoundError(f"画像が読み込めません: {image_path}")
        _rgb_uint8 = cv2.cvtColor(_bgr, cv2.COLOR_BGR2RGB)
        moge_input = torch.tensor(_rgb_uint8 / 255, dtype=torch.float32, device=device).permute(2, 0, 1)

        with torch.no_grad():
            moge_output = moge_model.infer(moge_input)

        # moge_output のキー: "points"(H,W,3) "depth"(H,W) "mask"(H,W) "normal"(H,W,3, optional) "intrinsics"(3,3)
        depth_map = moge_output["depth"].cpu().numpy()
        depth_mask = moge_output["mask"].cpu().numpy()
        points_map = moge_output["points"].cpu().numpy()
        intrinsics = moge_output["intrinsics"].cpu().numpy()
        normal_map = moge_output["normal"].cpu().numpy() if "normal" in moge_output else None

        print("depth map shape:", depth_map.shape)
        print(f"有効画素(mask=True)の割合: {100 * depth_mask.mean():.1f}%")

        # 結果を保存(Section 12.5・12.6が読み込む)
        depth_dir = os.path.join(WORK_DIR, "depth")
        os.makedirs(depth_dir, exist_ok=True)
        depth_npz_path = os.path.join(depth_dir, "moge_depth.npz")
        save_kwargs = dict(depth=depth_map, mask=depth_mask, points=points_map, intrinsics=intrinsics)
        if normal_map is not None:
            save_kwargs["normal"] = normal_map
        np.savez(depth_npz_path, **save_kwargs)
        print("saved:", depth_npz_path)


        # ---- detect_prompts (元ノートブック cell 23) ----
        # HSSD版DBにはカテゴリラベルが無いため、LVISカテゴリ一覧の表示は行わない
        # (以前はここでDB内のLVISカテゴリ一覧を表示し、プロンプト決めの参考にしていた)。

        # ==========================================
        # OmDet-Turboに渡す検出用プロンプト。自由に編集してよく、DBのLVISカテゴリ名と
        # 一致している必要はない(一致しない分は、Section 11でCLIPの類似度をもとに自動で
        # 対応するLVISカテゴリへ紐付けられる)。認識されやすい一般的な単語がおすすめ。
        DETECTION_PROMPTS = [
            # 座る家具
            "chair", "sofa", "loveseat", "futon",
            "bean bag chair", "stool",

            # テーブル類
            "table", "desk", "coffee table", "dining table",

            # 寝具
            "bed",

            # 収納家具
            "bookshelf", "cabinet", "dresser",

            # 台・スタンド類
            "tv stand", "coat rack",

            # キッチン家電
            "refrigerator", "oven", "microwave",

            # ランドリー
            "washing machine",

            # 照明
            "floor lamp", "table lamp",

            # 電子機器
            "laptop", "computer monitor","tv",

            # その他の設置物
            "potted plant", "trash can", "vacuum cleaner"
        ]
        # ==========================================

        print("\n検出器に渡すプロンプト:")
        print(DETECTION_PROMPTS)


        progress({"percent": _PCT["detect_sam3"], "label": "家具検出中"})

        # ---- detect_sam3 (元ノートブック cell 25) ----
        def _box_to_mask(image, box, processor, model, device):
            # 検出器のボックスをSAM3 Trackerに渡し、そのボックス領域の精密なマスクを得る。
            # SAM3 Trackerはボックスを「その領域をマスク化しろ」という指示として扱う
            # (SAM3の`Sam3Model`とは違い、ボックスを"お手本(exemplar)"として画像全体を
            # 再検索したりはしない)。
            inputs = processor(images=image, input_boxes=[[box]], return_tensors="pt").to(device)
            with torch.no_grad(), bf16_autocast():
                outputs = model(**inputs)

            masks = processor.post_process_masks(
                outputs.pred_masks.cpu(), inputs["original_sizes"].cpu()
            )
            mask_candidates = masks[0][0]                    # (num_masks, H, W)
            iou_scores = outputs.iou_scores.cpu()[0][0]      # (num_masks,)
            best_idx = int(iou_scores.argmax())
            return mask_candidates[best_idx].numpy()


        # ==========================================
        # image_path は Section 1.5(MoGe-2による深度推定)で既に定義済みのものをそのまま使う
        CATEGORY_CHUNK_SIZE = len(DETECTION_PROMPTS)  # 1回の検出器呼び出しでまとめて問い合わせるプロンプト数
        BOX_THRESHOLD = 0.30      # 下げるほど取りこぼしが減る(誤検出は増える)
        NMS_THRESHOLD = 0.5       # 同じクラス内の重複ボックスを間引くしきい値
        # ==========================================

        image = Image.open(image_path).convert("RGB")
        print(f"image size: {image.size}")

        detections = []  # 各要素: {"category": 検出プロンプト, "prompt": 同左, "mask":..., "box":..., "score":...}
        per_prompt_found = Counter()

        for i in range(0, len(DETECTION_PROMPTS), CATEGORY_CHUNK_SIZE):
            chunk = DETECTION_PROMPTS[i:i + CATEGORY_CHUNK_SIZE]
            det_inputs = omdet_processor(images=image, text=chunk, return_tensors="pt").to(device)
            with torch.no_grad(), bf16_autocast():
                det_outputs = omdet_model(**det_inputs)

            det_results = omdet_processor.post_process_grounded_object_detection(
                det_outputs,
                text_labels=chunk,
                threshold=BOX_THRESHOLD,
                nms_threshold=NMS_THRESHOLD,
                target_sizes=[(image.height, image.width)],
            )[0]

            for box, score, text_label in zip(det_results["boxes"], det_results["scores"], det_results["text_labels"]):
                per_prompt_found[text_label] += 1
                _item_t0 = time.time()

                box = [float(v) for v in box]
                mask = _box_to_mask(image, box, sam3_tracker_processor, sam3_tracker_model, device)
                _item_elapsed = time.time() - _item_t0
                print(f"    \u23f1\ufe0f '{text_label}' (score={float(score):.2f}) の検出+SAM3マスク化: {_item_elapsed:.2f}秒")
                # "category" にはDETECTION_PROMPTSの文字列(例: "chair")をそのまま入れておく。
                # これを実際のLVISカテゴリ(例: "office_chair")に対応付ける処理はSection 11で行う。
                detections.append({
                    "category": text_label,
                    "prompt": text_label,
                    "mask": mask,
                    "box": box,
                    "score": float(score),
                })

        for prompt in DETECTION_PROMPTS:
            print(f"'{prompt}': {per_prompt_found.get(prompt, 0)} 件検出")

        print(f"\n合計検出数: {len(detections)}")

        # --- 床(floor)も同様にOmDet-Turbo + SAM3 Trackerで検出・マスク化しておく。
        #     DETECTION_PROMPTSには含めない(家具として配置したいわけではなく、Section 12.5の
        #     ワールド整列で「床面の法線」を求めるためのマスクだけが欲しいため)。
        #     モデル(omdet_model・sam3_tracker_model)はこの直後のSection 7で解放されて
        #     しまうので、その前のここで検出しておく必要がある。 ---
        FLOOR_PROMPTS = ["floor", "flooring"]
        FLOOR_BOX_THRESHOLD = 0.15  # 床は輪郭が曖昧で検出しづらいため、家具用よりゆるいしきい値にする

        floor_mask_2d = None
        _floor_det_inputs = omdet_processor(images=image, text=FLOOR_PROMPTS, return_tensors="pt").to(device)
        with torch.no_grad(), bf16_autocast():
            _floor_det_outputs = omdet_model(**_floor_det_inputs)
        _floor_det_results = omdet_processor.post_process_grounded_object_detection(
            _floor_det_outputs,
            text_labels=FLOOR_PROMPTS,
            threshold=FLOOR_BOX_THRESHOLD,
            nms_threshold=NMS_THRESHOLD,
            target_sizes=[(image.height, image.width)],
        )[0]

        if len(_floor_det_results["boxes"]) > 0:
            _floor_scores = [float(s) for s in _floor_det_results["scores"]]
            _best_floor_idx = int(np.argmax(_floor_scores))
            _floor_box = [float(v) for v in _floor_det_results["boxes"][_best_floor_idx]]
            _floor_score = _floor_scores[_best_floor_idx]
            _floor_label = _floor_det_results["text_labels"][_best_floor_idx]
            floor_mask_2d = _box_to_mask(image, _floor_box, sam3_tracker_processor, sam3_tracker_model, device)
            _floor_mask_path = os.path.join(WORK_DIR, "floor_mask.npy")
            np.save(_floor_mask_path, floor_mask_2d)
            print(f"床を検出しました('{_floor_label}', score={_floor_score:.2f}, "
                  f"画素数={int(floor_mask_2d.sum())})。Section 12.5のワールド整列に使用します。"
                  f" saved: {_floor_mask_path}")
        else:
            print("床を検出できませんでした。Section 12.5のワールド整列は別の手法にフォールバックします。")


        # ---- dedup (元ノートブック cell 29) ----
        # ==========================================
        IOU_DEDUP_THRESHOLD = 0.5
        # ==========================================


        def compute_iou(box_a, box_b):
            xA = max(box_a[0], box_b[0])
            yA = max(box_a[1], box_b[1])
            xB = min(box_a[2], box_b[2])
            yB = min(box_a[3], box_b[3])
            inter_w = max(0.0, xB - xA)
            inter_h = max(0.0, yB - yA)
            inter_area = inter_w * inter_h
            area_a = max(0.0, box_a[2] - box_a[0]) * max(0.0, box_a[3] - box_a[1])
            area_b = max(0.0, box_b[2] - box_b[0]) * max(0.0, box_b[3] - box_b[1])
            union = area_a + area_b - inter_area
            return inter_area / union if union > 0 else 0.0


        n_before = len(detections)

        sorted_detections = sorted(detections, key=lambda d: -d["score"])
        deduped = []
        for det in sorted_detections:
            # 既に採用済みの検出とマスク(ボックス)が重なっているかを調べる。
            # 重なっていた場合、そのボックス自体は捨てる(1つに絞る)が、カテゴリ名は
            # 採用済みの検出の"候補名リスト"に加えて残しておく(検索時に複数名で検索するため)。
            matched_kept = None
            for kept in deduped:
                if compute_iou(det["box"], kept["box"]) > IOU_DEDUP_THRESHOLD:
                    matched_kept = kept
                    break

            if matched_kept is None:
                det["alt_categories"] = [det["category"]]
                deduped.append(det)
            else:
                if det["category"] not in matched_kept["alt_categories"]:
                    matched_kept["alt_categories"].append(det["category"])
                print(f"  重複としてマスクは統合(名前は候補として保持): "
                      f"{det['category']} (score={det['score']:.3f}) -> {matched_kept['category']}側へ")

        detections = deduped
        print(f"\n重複除去: {n_before}件 → {len(detections)}件")
        for det in detections:
            if len(det["alt_categories"]) > 1:
                print(f"  '{det['category']}' の候補名: {det['alt_categories']}")


        progress({"percent": _PCT["wilddet3d"], "label": "3D形状推定中"})

        # ---- wilddet3d (元ノートブック cell 32) ----
        # --- MoGe-2の正規化intrinsics(Section 1.5)をピクセル単位に変換 ---
        # MoGe-2のintrinsicsはutils3dの規約で正規化されている(0.5px分のオフセットは無視できるレベル
        # なので、ここでは fx = K[0,0]*W, fy = K[1,1]*H, cx = K[0,2]*W, cy = K[1,2]*H で近似する)。
        _h, _w = depth_map.shape
        pixel_intrinsics = intrinsics.copy().astype(np.float64)
        pixel_intrinsics[0, 0] *= _w  # fx
        pixel_intrinsics[1, 1] *= _h  # fy
        pixel_intrinsics[0, 2] *= _w  # cx
        pixel_intrinsics[1, 2] *= _h  # cy
        print("推定カメラ内部パラメータ(ピクセル単位):")
        print(pixel_intrinsics)

        # --- 前処理: 画像 + カメラ内部パラメータ + MoGe-2の深度マップ ---
        _depth_for_wilddet3d = np.nan_to_num(depth_map, nan=0.0).astype(np.float32)
        wilddet3d_data = preprocess(
            _rgb_uint8.astype(np.float32),
            intrinsics=pixel_intrinsics.astype(np.float32),
            depth=_depth_for_wilddet3d,
        )


        def _quaternion_wxyz_to_matrix(quaternion_wxyz):
            return _tf.quaternion_matrix(np.array(quaternion_wxyz))[:3, :3]


        def _obb_min_camera_distance(center_xyz, size_whl, quaternion_wxyz):
            """OBBの8頂点をカメラ座標系で求め、カメラ(原点)から見て一番近い頂点までの
            距離を返す(=このOBBのパラメータが「予測する」手前面までの距離)。
            Section 7.6の_obb_corners_camと同じ頂点並び・軸の対応(ローカルX軸=長さl、
            Y軸=高さh、Z軸=幅w)を使っている。"""
            w, h, l = size_whl
            half = np.array([l / 2.0, h / 2.0, w / 2.0])
            signs = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])
            R = _quaternion_wxyz_to_matrix(quaternion_wxyz)
            corners = (signs * half) @ R.T + np.array(center_xyz)
            return float(np.linalg.norm(corners, axis=1).min())


        def _observed_near_surface_distance(mask, points_map_, depth_mask_,
                                             min_points=OBB_DEPTH_CORRECTION_MIN_POINTS,
                                             iqr_mult=OBB_DEPTH_CORRECTION_MAX_IQR_MULT,
                                             percentile=OBB_DEPTH_CORRECTION_PERCENTILE):
            """物体のマスク領域に対応する深度点群から、外れ値(主にマスクが背景や隣の
            物体にはみ出して混ざった点)を除いたうえで、「手前面」の代表距離(低めの
            分位点)を求める。有効な点が少なすぎる場合はNoneを返す。"""
            valid = (mask > 0.5) & depth_mask_
            if int(valid.sum()) < min_points:
                return None
            pts = points_map_[valid]
            dists = np.linalg.norm(pts, axis=1)
            q1, q3 = np.percentile(dists, [25, 75])
            iqr = max(q3 - q1, 1e-6)
            lo, hi = q1 - iqr_mult * iqr, q3 + iqr_mult * iqr
            filtered = dists[(dists >= lo) & (dists <= hi)]
            if len(filtered) < max(min_points // 2, 5):
                filtered = dists  # 外れ値除去で減りすぎたら諦めて元の点群をそのまま使う
            return float(np.percentile(filtered, percentile))


        def _correct_obb_center_by_depth(obb_dict, mask, points_map_, depth_mask_, gain=OBB_DEPTH_CORRECTION_GAIN):
            """観測された深度点群の手前面距離と、現在のOBBが予測する手前面距離との差分
            だけ、OBBの中心を「カメラ→中心」の方向ベクトルに沿ってスライドさせる。
            方向はWildDet3Dの推定のまま変えず、距離だけを深度点群に合わせて調整する
            イメージ。gain(0〜1)で補正の強さを調整できる。計算できない場合はNoneを返す
            (呼び出し側は元のcenter_xyzをそのまま使う)。"""
            center = np.array(obb_dict["center_xyz"], dtype=np.float64)
            center_dist = np.linalg.norm(center)
            if center_dist < 1e-6:
                return None

            observed = _observed_near_surface_distance(mask, points_map_, depth_mask_)
            if observed is None:
                return None

            predicted = _obb_min_camera_distance(center, obb_dict["size_whl"], obb_dict["quaternion_wxyz"])
            delta = (observed - predicted) * gain

            direction = center / center_dist
            return center + direction * delta


        def wilddet3d_geometric_batch(predictor, images, intrinsics_t, input_hw, original_hw, padding,
                                       input_boxes, prompt_text="geometric", depth_gt=None):
            """複数の2DボックスプロンプトをSAM3画像エンコード1回にまとめて処理し、
            各ボックスに対応する3D OBBをボックスごとのIoUマッチングで個別に取り出す。

            WildDet3DPredictor.forward()のgeometricモード抽出ロジック(IoUトップ10 → 最高スコア)を、
            「画像ごと」ではなく「ボックスごと」にループするよう書き直したもの。画像は1枚
            (B_images=1)専用で、その1枚に対する全ボックスをimg_ids=0でまとめて渡す。
            """
            device = images.device
            H, W = input_hw
            n_prompts = len(input_boxes)

            # 各ボックスを元画像座標 -> モデル入力(リサイズ+パディング済み)座標へ変換
            input_boxes_model = [
                _orig_to_input_hw_box(box, original_hw, padding, (H, W)) for box in input_boxes
            ]
            boxes_cxcywh = []
            for x1, y1, x2, y2 in input_boxes_model:
                boxes_cxcywh.append([(x1 + x2) / 2 / W, (y1 + y2) / 2 / H, (x2 - x1) / W, (y2 - y1) / H])
            geo_boxes = torch.tensor(boxes_cxcywh, dtype=torch.float32, device=device).unsqueeze(1)  # (N,1,4)

            batch = WildDet3DInput(
                images=images,
                intrinsics=intrinsics_t,
                img_ids=torch.zeros(n_prompts, dtype=torch.long, device=device),   # 全ボックスを同じ画像0に紐付け
                text_ids=torch.zeros(n_prompts, dtype=torch.long, device=device),
                unique_texts=[prompt_text],
                geo_boxes=geo_boxes,
                geo_boxes_mask=torch.zeros(n_prompts, 1, dtype=torch.bool, device=device),
                geo_box_labels=torch.ones(n_prompts, 1, dtype=torch.long, device=device),
                original_hw=[original_hw],  # 指定しておくとモデル内部で元画像座標へ自動リスケールしてくれる
                padding=[padding],
                depth_gt=depth_gt,
            )

            with torch.no_grad(), bf16_autocast():
                # WildDet3DPredictor.forward()(公式ラッパー)を経由せず、内部モデルを直接呼ぶ。
                # SAM3の画像エンコード(self.sam3.backbone.forward_image)は batch.images
                # (=1枚)に対して1回だけ実行され、n_prompts個のボックスはその共有された
                # 画像特徴量に対してまとめてデコードされる。
                output = predictor.wilddet3d(batch)

            cand_boxes = output.boxes[0]        # (M, 4) 元画像pixel空間、全ボックスぶんの候補プール
            cand_boxes3d = output.boxes3d[0]    # (M, 10)
            cand_scores = output.scores[0]
            cand_scores_2d = output.scores_2d[0] if output.scores_2d is not None else cand_scores
            cand_scores_3d = (
                output.scores_3d[0] if output.scores_3d is not None else torch.zeros_like(cand_scores)
            )

            results = []
            for box in input_boxes:
                if cand_boxes.shape[0] == 0:
                    results.append(None)
                    continue
                prompt_box = torch.tensor(box, dtype=torch.float32, device=cand_boxes.device)
                ious = _pairwise_iou(cand_boxes, prompt_box.unsqueeze(0)).squeeze(-1)
                if ious.numel() == 0 or ious.max() <= 0:
                    best = int(cand_scores.argmax())
                else:
                    topk = min(10, ious.numel())
                    _, topk_idx = ious.topk(topk)
                    best = int(topk_idx[cand_scores[topk_idx].argmax()])
                results.append({
                    "obb_3d": cand_boxes3d[best].detach().float().cpu().numpy(),
                    "score_2d": float(cand_scores_2d[best]),
                    "score_3d": float(cand_scores_3d[best]),
                })
            return results


        # --- OmDet-Turboで得た家具ごとの2Dボックス(Section 7で重複除去済み)を、
        #     SAM3画像エンコード1回にまとめてWildDet3Dに渡し、3D OBBを一括取得する ---
        all_boxes_xyxy = [[float(v) for v in det["box"]] for det in detections]

        _batch_t0 = time.time()
        batch_results = wilddet3d_geometric_batch(
            wilddet3d_model,
            images=wilddet3d_data["images"].to(device),
            intrinsics_t=wilddet3d_data["intrinsics"].to(device)[None],
            input_hw=wilddet3d_data["input_hw"],
            original_hw=wilddet3d_data["original_hw"],
            padding=wilddet3d_data["padding"],
            input_boxes=all_boxes_xyxy,
            prompt_text="geometric",
            depth_gt=wilddet3d_data["depth_gt"].to(device),
        )
        _batch_elapsed = time.time() - _batch_t0
        _avg = _batch_elapsed / max(1, len(detections))
        print(f"  \u23f1\ufe0f {len(detections)}件のボックスをまとめて1回のforwardで処理: "
              f"{_batch_elapsed:.2f}秒(1件あたり平均 {_avg:.3f}秒)")

        n_obb_ok = 0
        for det, res in zip(detections, batch_results):
            _item_t0 = time.time()
            if res is None:
                print(f"  [WARN] OBBが得られませんでした: {det['category']} box={det['box']}")
                det["obb_3d"] = None
                continue

            # 注意: WildDet3D(vis4d)の生ベクトルは (10,) = [x, y, z, w, l, h, qw, qx, qy, qz]
            # という並び(中心3 + サイズ3 + クォータニオン4)。サイズはw(幅),l(長さ),h(高さ)の順で
            # あり、"w,h,l"ではない(vis4d/op/box/box3d.pyの
            # `w, l, h = boxes3d[:, 3], boxes3d[:, 4], boxes3d[:, 5]` で確認済み)。
            # 以前はindex4を高さ・index5を奥行きとして扱っており、高さと奥行き(長さ)を
            # 取り違えていた(3D配置で軸が完全におかしくなる原因だった)。
            # ここで size_whl=[幅, 高さ, 奥行き] の順に並べ直してから保存し、以降(Section 10.3の
            # サイズフィルタ、Section 12.5の3D配置)のコードは変更しなくて済むようにする。
            obb = res["obb_3d"]  # (10,) = x,y,z,w,l,h,qw,qx,qy,qz
            _w, _l, _h = obb[3], obb[4], obb[5]
            det["obb_3d"] = {
                "center_xyz": obb[0:3].tolist(),
                "size_whl": [float(_w), float(_h), float(_l)],  # [幅, 高さ, 奥行き] の順に並べ直す
                "quaternion_wxyz": obb[6:10].tolist(),
                "score_2d": res["score_2d"],
                "score_3d": res["score_3d"],
            }

            # --- 深度点群を使って、OBBの奥行き(カメラからの距離)だけを補正する ---
            if OBB_DEPTH_CORRECTION_ENABLED:
                _corrected_center = _correct_obb_center_by_depth(
                    det["obb_3d"], det["mask"], points_map, depth_mask
                )
                if _corrected_center is not None:
                    _old_center = np.array(det["obb_3d"]["center_xyz"])
                    _shift = float(np.linalg.norm(_corrected_center - _old_center))
                    det["obb_3d"]["center_xyz"] = _corrected_center.tolist()
                    if _shift > 0.01:
                        print(f"    [depth補正] '{det['category']}': 奥行き方向に{_shift:.3f}m補正しました")
                else:
                    print(f"    [depth補正] '{det['category']}': 有効な深度点が少なすぎるため補正をスキップ")

            n_obb_ok += 1
            _item_elapsed = time.time() - _item_t0
            print(f"    \u23f1\ufe0f '{det['category']}' のOBB抽出(IoUマッチングのみ、画像エンコードは共有済み): "
                  f"{_item_elapsed:.4f}秒")

        print(f"\nOBB取得: {n_obb_ok} / {len(detections)} 件")

        # --- 結果の保存 ---
        obb_dir = os.path.join(WORK_DIR, "obb_3d")
        os.makedirs(obb_dir, exist_ok=True)
        obb_json_path = os.path.join(obb_dir, "furniture_obb3d.json")
        with open(obb_json_path, "w", encoding="utf-8") as f:
            json.dump(
                [
                    {"category": det["category"], "box": det["box"], "score": det["score"], "obb_3d": det["obb_3d"]}
                    for det in detections
                ],
                f, ensure_ascii=False, indent=2,
            )
        print("saved:", obb_json_path)

        # --- 可視化: 全家具のOBBをまとめて元画像に重ねて描画 ---
        _valid_obbs = [det["obb_3d"] for det in detections if det["obb_3d"] is not None]
        if _valid_obbs:
            # 注意: draw_3d_boxes(WildDet3D公式)は生の並び(w, l, h)を前提にしている。
            # size_whlはこちら側の下流コード(Section 10.3/12.5)向けに[幅, 高さ, 奥行き]=(w, h, l)
            # の順に並べ替え済みなので、可視化のときだけ(w, l, h)に戻してから渡す
            # (そのまま渡すと高さと奥行きが入れ替わった箱が描かれてしまう)。
            _boxes3d_all = torch.tensor(
                [
                    obb["center_xyz"]
                    + [obb["size_whl"][0], obb["size_whl"][2], obb["size_whl"][1]]  # (w,h,l) -> (w,l,h) に戻す
                    + obb["quaternion_wxyz"]
                    for obb in _valid_obbs
                ],
                dtype=torch.float32,
            )
            _scores2d_all = torch.tensor([obb["score_2d"] for obb in _valid_obbs], dtype=torch.float32)
            _scores3d_all = torch.tensor([obb["score_3d"] for obb in _valid_obbs], dtype=torch.float32)
            _class_names_all = [det["category"] for det in detections if det["obb_3d"] is not None]
            _class_ids_all = torch.arange(len(_class_names_all))

            pass
        else:
            print("有効なOBBが1件もありませんでした。")


        # ---- crop_save (元ノートブック cell 34) ----
        output_dir = os.path.join(WORK_DIR, "extracted_objects")
        mask_dir = os.path.join(output_dir, "masks")
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(mask_dir, exist_ok=True)

        # ==========================================
        BACKGROUND_COLOR = (128, 128, 128)  # マスク外を塗りつぶす色(中間グレー)
        # 真っ白だと、白っぽい家具が背景に溶けて見づらくなるため、明るい家具・暗い家具の
        # どちらともある程度コントラストがつきやすい中間グレーにしている。
        # ==========================================

        per_category_counter = {}
        metadata = []

        img_w, img_h = image.size
        img_array = np.array(image.convert("RGB"))

        for det in detections:
            cat = det["category"]
            idx = per_category_counter.get(cat, 0)
            per_category_counter[cat] = idx + 1
            _item_t0 = time.time()

            x1, y1, x2, y2 = det["box"]
            x1 = max(0, min(img_w, x1))
            x2 = max(0, min(img_w, x2))
            y1 = max(0, min(img_h, y1))
            y2 = max(0, min(img_h, y2))
            if x2 <= x1 or y2 <= y1:
                print(f"skip invalid box: {det['box']} ({cat})")
                continue

            mask = det["mask"].astype(bool)

            # マスク外(背景・隣接する別の物体など)を単色で塗りつぶしてからクロップする。
            # こうすることで、Section 10のCLIP画像埋め込みに余計な背景情報が混ざるのを防げる。
            # masked_array = img_array.copy()
            # masked_array[~mask] = BACKGROUND_COLOR
            # masked_image = Image.fromarray(masked_array)

            # crop = masked_image.crop((x1, y1, x2, y2))

            crop = image.crop((x1, y1, x2, y2))

            out_name = f"{cat}_{idx}.png"
            out_path = os.path.join(output_dir, out_name)
            crop.save(out_path)

            # マスク画像(元画像と同じサイズ、マスク部分=白(255)、それ以外=黒(0)の2値画像)を保存
            mask_array = (mask.astype(np.uint8)) * 255
            mask_image = Image.fromarray(mask_array, mode="L")
            mask_name = f"{cat}_{idx}_mask.png"
            mask_path = os.path.join(mask_dir, mask_name)
            mask_image.save(mask_path)

            # WildDet3D(Section 7.5)で計算済みのOBB(位置・姿勢・サイズ)をそのまま引き継ぐ
            # (Section 10.3のサイズ絞り込み検索、Section 12.5の3D配置で使う。取得失敗時はNone)
            _obb = det.get("obb_3d")
            size_whl = _obb["size_whl"] if _obb else None

            metadata.append({
                "file": out_name,
                "mask_file": os.path.join("masks", mask_name),
                "category": cat,  # 代表カテゴリ(スコアが最も高かったプロンプト名。表示用)
                "alt_categories": det.get("alt_categories", [cat]),  # Section 7で重複統合された全候補名
                "box": [x1, y1, x2, y2],
                "score": det["score"],
                "size_whl": size_whl,  # WildDet3DのOBB由来。[幅, 高さ, 奥行き](メートル)、取得失敗時はNone
                "obb_3d": _obb,  # center_xyz・size_whl・quaternion_wxyzを含む完全な情報(3D配置用)
            })
            print(f"    \u23f1\ufe0f '{out_name}' の切り出し保存: {time.time() - _item_t0:.2f}秒")

        with open(os.path.join(output_dir, "metadata.json"), "w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)

        print(f"{len(metadata)}件を {output_dir} に保存しました(マスク画像は {mask_dir} 配下)。")


        progress({"percent": _PCT["clip_map"], "label": "類似モデル検索中"})

        # ---- clip_map (元ノートブック cell 38) ----

        def _text_embeddings(phrases, clip_model, clip_tokenizer, device):
            # Section 10.3(ハイブリッド検索)でも使う、CLIPのテキストエンコーダのヘルパー関数。
            # 以前はここ(Section 10.2)でprompt->LVISカテゴリの対応付けにも使っていたが、
            # HSSD版DBにはカテゴリが無いためその用途は無くなった。関数自体はSection 10.3から
            # 参照されるため、ここに残しておく。
            with torch.no_grad(), bf16_autocast():
                tokens = clip_tokenizer(phrases).to(device)
                embeds = clip_model.encode_text(tokens)
                embeds = embeds / embeds.norm(dim=-1, keepdim=True)
            return embeds.float().cpu().numpy()


        # HSSD版DBにはカテゴリが無いため、常に「絞り込みなし」(DB全体を検索対象にする)。
        # (以前はここでCLIPのテキスト類似度を使ってprompt->LVISカテゴリの対応表を作っていた)
        prompt_to_categories = {prompt: None for prompt in DETECTION_PROMPTS}
        print("カテゴリによる絞り込みは無効化されています(HSSD版DBにはカテゴリラベルが無いため)。")
        print("すべてのプロンプトでDB全体を検索対象にします。")


        # ---- clip_search (元ノートブック cell 40) ----
        # ==========================================
        # 検索は3段階のパイプラインにしている:
        #   1. テキスト(候補名すべて)で広く候補を集める(上位TEXT_STAGE_TOP_N件)
        #   2. 実寸サイズ(WildDet3DのOBB)で「明らかに論外」なものだけを緩く除外する
        #   3. 残った候補を「画像類似度 + サイズの近さ」の合成スコアで最終ランキングし、
        #      上位TOP_K件を返す
        #
        # (以前はStage 2を厳しめのハードフィルタにしていたが、絞り込みすぎると
        #  「たまたまサイズだけ合った、見た目は無関係な候補」しか残らないことがあった。
        #  3Dモデルはどうせ後でスケールを合わせて配置するので、サイズは「完全一致必須」
        #  ではなく「近い方が望ましい」程度の弱い判断材料にする方が実用的、という判断で変更した)
        TOP_K = 5
        TEXT_STAGE_TOP_N = 300     # 段階1(テキスト)で残す候補数
        ENABLE_SIZE_FILTER = True  # WildDet3DのOBBサイズを判断材料に使うかどうか
        MAX_SIZE_RATIO = 3.0       # Stage 2(緩いハードフィルタ)の閾値。「明らかに論外」なサイズ
                                    # (例: 一桁違う)だけを除外する目的なので、以前よりだいぶ緩め。
        SIZE_FILTER_FALLBACK_IF_EMPTY = True  # 絞り込みすぎて0件になった場合に、フィルタ無しへ
                                    # フォールバックするか。
        SIZE_SCORE_GAMMA = 0.7     # Stage 3の最終ランキングでの重み(画像類似度の重み。
                                    # 1-SIZE_SCORE_GAMMA がサイズの近さの重み。1.0にすると
                                    # 以前と同じ「画像類似度のみ」の挙動に戻る)
        # ==========================================

        _required = {
            "metadata": "Section 8(インスタンスごとに切り出して保存)",
            "output_dir": "Section 8(インスタンスごとに切り出して保存)",
            "db_embeddings": "Section 4.4(対象UIDの埋め込み抽出)",
            "db_names": "Section 4.4(対象UIDの埋め込み抽出)",
            "db_categories": "Section 4.4(対象UIDの埋め込み抽出)",
            "clip_model": "Section 10.1(CLIPモデルをロードする)",
            "clip_tokenizer": "Section 10.1(CLIPモデルをロードする)",
            "clip_preprocess": "Section 10.1(CLIPモデルをロードする)",
            "prompt_to_categories": "Section 10.2(検出プロンプト -> LVISカテゴリの対応表)",
        }
        # 注意: list内包表記の中でlocals()を呼ぶと、その内包表記自身の小さな
        # スコープ(ループ変数だけ)しか見えないため、先に外側で名前空間を確定させておく。
        _known_names = set(locals().keys()) | set(globals().keys())
        _missing = [(name, sec) for name, sec in _required.items() if name not in _known_names]
        if _missing:
            msg = "\n".join(f"  - {name} が未定義 -> {sec} を先に実行してください" for name, sec in _missing)
            raise NameError(
                "検索に必要な変数が揃っていません。カーネル再起動後や、セルを飛ばして実行した場合に"
                f"起こります。以下を確認してください:\n{msg}\n"
                "確実なのは Kernel > Restart & Run All で最初から通しで実行することです。"
            )


        def _apply_size_filter(indices, query_size_whl, db_bbox_whd, max_ratio,
                                fallback_if_empty=True, verbose=False):
            """クエリのOBBサイズ(WildDet3D由来、[幅, 高さ, 奥行き]メートル)と、DB候補の
            バウンディングボックス寸法(dx, dy, dz。dy=高さ)を比較し、サイズが大きく異なる
            候補をindicesから除外する。

            水平方向2軸(幅・奥行き)は、クエリ側とDB側で「どちらの軸がどちらに対応するか」が
            分からない(候補の向きがクエリと同じとは限らないため)ので、大小関係でソートしてから
            比較する(＝回転・軸の対応関係に依存しない比較にする)。

            サイズ情報が無い(query_size_whlがNone、db_bbox_whdがNone、または値がNaN/極端に小さい)
            場合は「判定不能」として絞り込み対象から外さない(＝安全側に倒す)。
            """
            if query_size_whl is None or db_bbox_whd is None:
                return indices

            q = np.asarray(query_size_whl, dtype=np.float64)
            if q.shape != (3,) or np.any(~np.isfinite(q)) or np.any(q <= 1e-3):
                return indices  # クエリ側のサイズが取得できていない/不正なら絞り込みしない

            sub = db_bbox_whd[indices]  # (N, 3) = (dx, dy, dz)
            valid = np.all(np.isfinite(sub), axis=1) & np.all(sub > 1e-3, axis=1)

            q_height = q[1]
            q_h_lo, q_h_hi = sorted([q[0], q[2]])  # 水平2方向を大小でソート(対応関係を気にしない)

            d_height = sub[:, 1]
            d_h_lo = np.minimum(sub[:, 0], sub[:, 2])
            d_h_hi = np.maximum(sub[:, 0], sub[:, 2])

            def _ratio(a, b):
                a = np.maximum(a, 1e-6)
                b = np.maximum(b, 1e-6)
                return np.maximum(a, b) / np.minimum(a, b)

            worst_ratio = np.maximum(
                np.maximum(_ratio(q_height, d_height), _ratio(q_h_lo, d_h_lo)),
                _ratio(q_h_hi, d_h_hi),
            )

            keep = (~valid) | (worst_ratio <= max_ratio)  # サイズ不明(invalid)なものは判定できないので残す
            filtered = indices[keep]

            if len(filtered) < len(indices) and verbose:
                print(f"    [size_filter] {len(indices)}件 -> {len(filtered)}件"
                      f"(除外 {len(indices) - len(filtered)}件, max_ratio={max_ratio})")

            # 絞り込みすぎて候補が0件になった場合、fallback_if_empty=Trueならフィルタなしに
            # 戻す(結果が消える事態を避ける)。Falseなら素直に0件を返す(絞り込みを完全に尊重)。
            # 注意: fallback_if_empty=Trueのままmax_ratioを極端に厳しくすると(例:1に近い値)、
            # 常に0件→常にフォールバック、という状態になり「フィルタが効いていないように見える」
            # ことがある。フィルタの効果を確認したい場合は fallback_if_empty=False にすること。
            if len(filtered) == 0:
                if fallback_if_empty:
                    if verbose:
                        print("    [size_filter] 絞り込みで0件になったため、フィルタ無しにフォールバックしました")
                    return indices
                return filtered
            return filtered


        def _size_similarity_scores(query_size_whl, db_bbox_whd, indices, neutral_score=0.5):
            """DB候補ごとに「サイズの近さ」を0〜1のスコア(1に近いほど良い)で返す。
            _apply_size_filter と同じ比較ロジック(高さ・横2方向をソートして比較)を使うが、
            こちらは足切りせず、1/worst_ratio という連続値にする(worst_ratio=1=完全一致で
            スコア1.0、ズレが大きいほど0に近づく)。

            サイズ情報が無い(query_size_whlがNone、db_bbox_whdがNone、または対象候補の値が
            NaN/極端に小さい)場合は、有利にも不利にもしないよう neutral_score(既定0.5)を返す。
            """
            n = len(indices)
            if query_size_whl is None or db_bbox_whd is None:
                return np.full(n, neutral_score, dtype=np.float64)

            q = np.asarray(query_size_whl, dtype=np.float64)
            if q.shape != (3,) or np.any(~np.isfinite(q)) or np.any(q <= 1e-3):
                return np.full(n, neutral_score, dtype=np.float64)

            sub = db_bbox_whd[indices]  # (N, 3) = (dx, dy, dz)
            valid = np.all(np.isfinite(sub), axis=1) & np.all(sub > 1e-3, axis=1)

            q_height = q[1]
            q_h_lo, q_h_hi = sorted([q[0], q[2]])

            d_height = sub[:, 1]
            d_h_lo = np.minimum(sub[:, 0], sub[:, 2])
            d_h_hi = np.maximum(sub[:, 0], sub[:, 2])

            def _ratio(a, b):
                a = np.maximum(a, 1e-6)
                b = np.maximum(b, 1e-6)
                return np.maximum(a, b) / np.minimum(a, b)

            worst_ratio = np.maximum(
                np.maximum(_ratio(q_height, d_height), _ratio(q_h_lo, d_h_lo)),
                _ratio(q_h_hi, d_h_hi),
            )
            size_score = 1.0 / worst_ratio
            size_score = np.where(valid, size_score, neutral_score)
            return size_score


        def _image_embedding(path, clip_model, clip_preprocess, device, mask_path=None, box=None,
                              background_color=None):
            """クロップ画像のCLIP画像埋め込みを計算する。

            mask_path(Section 8で保存したSAM3マスク、元画像と同じフルサイズの2値画像)と
            box(そのインスタンスの[x1,y1,x2,y2])が渡された場合、マスクをboxでクロップして
            クロップ画像に重ね、マスク外(背景・隣接する別の物体など)を単色で塗りつぶしてから
            CLIPに通す(「見た目」の類似度に余計な背景が混ざらないようにするため)。
            """
            img = Image.open(path).convert("RGB")

            if mask_path is not None and box is not None and os.path.exists(mask_path):
                if background_color is None:
                    background_color = globals().get("BACKGROUND_COLOR", (128, 128, 128))
                x1, y1, x2, y2 = [int(round(v)) for v in box]
                mask_full = Image.open(mask_path).convert("L")
                mask_crop = mask_full.crop((x1, y1, x2, y2))
                mask_arr = np.array(mask_crop) > 127
                img_arr = np.array(img)
                if mask_arr.shape[:2] == img_arr.shape[:2]:
                    img_arr = img_arr.copy()
                    img_arr[~mask_arr] = background_color
                    img = Image.fromarray(img_arr)
                else:
                    print(f"  [WARN] マスクとクロップのサイズが一致しないためマスク適用をスキップ: "
                          f"mask={mask_arr.shape[:2]} crop={img_arr.shape[:2]}")

            tensor = clip_preprocess(img).unsqueeze(0).to(device)
            with torch.no_grad():
                embed = clip_model.encode_image(tensor)
                embed = embed / embed.norm(dim=-1, keepdim=True)
            return embed.float().cpu().numpy()[0]


        def _search_db_hybrid(query_texts, image_path, top_k=TOP_K, category_filter=None,
                               mask_path=None, box=None, query_size_whl=None,
                               text_stage_top_n=TEXT_STAGE_TOP_N):
            """3段階の絞り込みパイプラインで検索する(関数名は互換性のため変えていないが、
            もはや「ハイブリッドスコアのブレンド」はしていない)。

            query_texts は文字列1個でも、文字列のリスト(重複除去で統合された複数の候補名。
            例: ["desk", "table"])でもよい。

            Stage 1 (テキスト): 候補名それぞれとのCLIPテキスト類似度を計算し、DB候補ごとに
              一番一致度の高い名前のスコアを採用したうえで、上位 text_stage_top_n 件に絞る。
            Stage 2 (サイズ・緩いフィルタ): WildDet3DのOBBサイズと比べて「明らかに論外」な
              サイズの候補だけを除外する(MAX_SIZE_RATIOは緩め)。
            Stage 3 (画像 + サイズの合成ランキング): 残った候補を、画像類似度とサイズの近さの
              重み付き合計(SIZE_SCORE_GAMMA)で並べ替え、上位 top_k 件を返す。3Dモデルは
              どうせ後でスケールを合わせて配置するため、サイズは強い足切り条件ではなく
              「近い方が望ましい」程度の弱い判断材料として最終順位に反映する。
            """
            if isinstance(query_texts, str):
                query_texts = [query_texts]
            text_embs = _text_embeddings(query_texts, clip_model, clip_tokenizer, device)  # (n_names, D)

            # Stage 0: カテゴリ絞り込み(HSSD版はdb_categoriesがNoneなので常にDB全体が対象)
            if category_filter and db_categories is not None:
                mask = np.isin(db_categories, category_filter)
                indices = np.where(mask)[0]
            else:
                indices = np.arange(len(db_embeddings))

            if len(indices) == 0:
                return []

            # --- Stage 1: テキスト類似度で広く候補を集める ---
            text_sims_per_name = db_embeddings[indices] @ text_embs.T  # (N, n_names)
            text_sims = text_sims_per_name.max(axis=1)  # 候補ごとに一番一致度の高い名前のスコアを採用
            stage1_n = min(text_stage_top_n, len(indices))
            stage1_order = np.argsort(-text_sims)[:stage1_n]
            stage1_indices = indices[stage1_order]

            # --- Stage 2: 実寸サイズで絞り込む ---
            if ENABLE_SIZE_FILTER and db_bbox_whd is not None:
                stage2_indices = _apply_size_filter(
                    stage1_indices, query_size_whl, db_bbox_whd, MAX_SIZE_RATIO,
                    fallback_if_empty=SIZE_FILTER_FALLBACK_IF_EMPTY, verbose=True,
                )
            else:
                stage2_indices = stage1_indices

            if len(stage2_indices) == 0:
                return []

            # --- Stage 3: 残った候補を「画像類似度 + サイズの近さ」の合成スコアで最終ランキング ---
            image_emb = _image_embedding(image_path, clip_model, clip_preprocess, device, mask_path=mask_path, box=box)
            image_sims = db_embeddings[stage2_indices] @ image_emb  # 両方L2正規化済みなので内積=コサイン類似度
            size_scores = (
                _size_similarity_scores(query_size_whl, db_bbox_whd, stage2_indices)
                if ENABLE_SIZE_FILTER and db_bbox_whd is not None
                else np.full(len(stage2_indices), 0.5, dtype=np.float64)
            )
            final_scores = SIZE_SCORE_GAMMA * image_sims + (1 - SIZE_SCORE_GAMMA) * size_scores

            k = min(top_k, len(stage2_indices))
            order = np.argsort(-final_scores)[:k]
            top_indices = stage2_indices[order]

            # 表示用に、最終候補に絞ってからテキスト類似度も計算し直す(件数が少ないので軽い)
            text_sims_final = (db_embeddings[top_indices] @ text_embs.T).max(axis=1)

            results = []
            for i, idx in enumerate(top_indices):
                item = {
                    "name": str(db_names[idx]),
                    "score": float(final_scores[order[i]]),   # 最終順位=画像類似度+サイズの近さの合成スコア
                    "image_score": float(image_sims[order[i]]),
                    "size_score": float(size_scores[order[i]]),
                    "text_score": float(text_sims_final[i]),
                }
                if db_categories is not None:
                    item["category"] = str(db_categories[idx])
                results.append(item)
            return results


        search_results = []  # 各要素: {"file":..., "category":..., "query":..., "candidates": [...]}

        for entry in metadata:
            category = entry["category"]  # OmDet-Turboが検出したプロンプト文字列(例: "chair"。代表/表示用)
            query_names = entry.get("alt_categories") or [category]  # 重複除去で統合された全候補名
            query = " / ".join(query_names)  # 表示用にまとめた文字列
            crop_path = os.path.join(output_dir, entry["file"])
            # SAM3マスク(Section 8で保存済み)を使い、画像側の類似度計算では背景を単色で塗りつぶした
            # 「見た目」を使う(box自体はマスクをクロップ範囲に合わせて切り出すために必要)
            mask_path = os.path.join(output_dir, entry["mask_file"]) if entry.get("mask_file") else None

            category_filter = prompt_to_categories.get(category)
            query_size_whl = entry.get("size_whl")  # WildDet3DのOBBサイズ(Section 8で引き継ぎ済み)

            _item_t0 = time.time()
            candidates = _search_db_hybrid(
                query_names, crop_path, top_k=TOP_K, category_filter=category_filter,
                mask_path=mask_path, box=entry["box"], query_size_whl=query_size_whl,
            )
            _item_elapsed = time.time() - _item_t0
            search_results.append({
                "file": entry["file"],
                "box": entry["box"],
                "mask_file": entry.get("mask_file"),
                "category": category,
                "query": query,
                "query_names": query_names,
                "candidates": candidates,
                "obb_3d": entry.get("obb_3d"),  # 3D配置(Section 12.5)で使う
            })

            if not candidates:
                print(f'skip: {entry["file"]} (query="{query}") のDB候補が0件 (処理時間 {_item_elapsed:.2f}秒)')
                continue

            filter_note = f" (絞り込み先カテゴリ: {category_filter})" if category_filter else " (絞り込みなし: DB全体)"
            print(f'{entry["file"]} (category={category}, query="{query}"){filter_note}:')
            for c in candidates:
                _cat_note = f"  category={c['category']}" if "category" in c else ""
                print(f"    {c['name']}{_cat_note}  score={c['score']:.4f}  "
                      f"(image={c['image_score']:.4f}, size={c['size_score']:.4f}, text={c['text_score']:.4f})")
            print(f"    \u23f1\ufe0f 検索処理: {_item_elapsed:.2f}秒")


        # ---- save_search (元ノートブック cell 42) ----
        search_results_path = os.path.join(WORK_DIR, "search_results.json")
        with open(search_results_path, "w", encoding="utf-8") as f:
            json.dump(search_results, f, ensure_ascii=False, indent=2)

        n_with_candidates = sum(1 for r in search_results if r["candidates"])
        print(f"検索対象: {len(search_results)}件中、候補が見つかったもの: {n_with_candidates}件")
        print("saved:", search_results_path)


        # ---- utils (元ノートブック cell 44) ----
        # --- Section 12.5・12.7・13・13.5 で使う共通ユーティリティ
        #     (以前は「12. 見つかった家具を表示する」セクションの一部だったが、
        #     プレビュー表示専用のセクションを削除したため、ここに独立させた) ---
        import base64

        HSSD_REPO_ID = "hssd/hssd-hab"
        _uid_to_relpath_map = dict(zip(db_names.tolist(), db_relpaths.tolist()))


        def _file_to_data_uri(bytes_data, mime):
            b64 = base64.b64encode(bytes_data).decode("ascii")
            return f"data:{mime};base64,{b64}"


        def _decimate_with_pyfqmr(vertices, faces, target_count, aggressiveness):
            # pyfqmrで頂点削減する。`preserve_border=True`が使え、開いた境界(座面・背もたれなど
            # 薄いパーツの縁)にある頂点を保護できるため、削減後に穴が開くのを防げる。
            simplifier = pyfqmr.Simplify()
            simplifier.setMesh(vertices, faces)
            simplifier.simplify_mesh(
                target_count=int(target_count),
                aggressiveness=aggressiveness,
                preserve_border=True,
                verbose=False,
            )
            new_vertices, new_faces, _normals = simplifier.getMesh()
            return new_vertices, new_faces


        progress({"percent": _PCT["placement"], "label": "家具配置中"})

        # ---- placement (元ノートブック cell 46) ----
        # ==========================================
        PLACEMENT_MAX_FACES = 20000       # 配置用メッシュの目標面数上限(表示用より高品質にしておく)
        PLACEMENT_DECIMATE_AGGRESSIVENESS = 5
        PLACEMENT_DOWNLOAD_WORKERS = 8
        PLACEMENT_DOWNLOAD_DIR = os.path.join(WORK_DIR, "_tmp_placement_objects")
        # --- 家具の色を、検出時の写真の色に近づける設定 ---
        # 注意: PLACEMENT_MAX_FACESを超えるメッシュは_decimate_with_pyfqmr()で簡略化されるが、
        # pyfqmrは頂点色・テクスチャを引き継がないため、そのままだとtrimeshの既定色(ほぼ白)に
        # なってしまう。これを防ぐため、デシメーション前に元の頂点色を保存しておき、
        # 簡略化後の頂点に最近傍の元頂点の色を引き継がせたうえで、色相を写真の色へ寄せる。
        RECOLOR_TO_MATCH_PHOTO = True      # Falseにすると、元モデルの色をそのまま使う(色合わせ無効)
        RECOLOR_HUE_STRENGTH = 0.85        # 色相(Hue)を写真の色へ寄せる強さ(0=元の色相のまま, 1=完全に一致)
        RECOLOR_SATURATION_STRENGTH = 0.5  # 彩度(Saturation)を写真の色へ寄せる強さ(0=元のまま, 1=完全に一致)
                                             # 明度(Value)は常に元のまま保持する(陰影・質感を残すため)
        RECOLOR_N_COLOR_CLUSTERS = 3       # 写真・モデルの色をそれぞれ何色にクラスタリングするか
                                             # (木の脚+ファブリックの座面、のような多色の家具に対応するため。
                                             # 1にすると、以前と同じ「全体を1色で塗る」動作になる)
        RECOLOR_MIN_TARGET_SATURATION = 0.12  # 写真側の色の彩度がこれ未満(白・黒・グレーに近い)なら、
                                             # 色相を無理に合わせない(彩度が低い色は色相の値自体が
                                             # 数値的に不安定なため)
        # ==========================================

        from concurrent.futures import ThreadPoolExecutor, as_completed
        from scipy.spatial import cKDTree
        from scipy.cluster.vq import kmeans2
        import matplotlib.colors as mcolors


        def _bake_vertex_colors(mesh):
            """meshの見た目(テクスチャ or 頂点色のどちらでも)を、頂点ごとのRGBA配列(uint8)として
            取り出す。"""
            return mesh.visual.to_color().vertex_colors.astype(np.uint8)


        def _extract_target_color_clusters(crop_path, mask_path, box, n_clusters=RECOLOR_N_COLOR_CLUSTERS):
            """検出時のクロップ画像(マスクで背景を塗りつぶし済み)から、家具本体部分
            (mask==True)の色をn_clusters色にクラスタリング(k-means)し、
            [(色RGB 0-255, 割合), ...]を割合の大きい順に返す(木の脚+ファブリックの座面、の
            ように複数の色を持つ家具に対応するため)。マスクが無い/使えない場合はクロップ
            全体を使う。ピクセル数が少なすぎる場合は色1つ(中央値)にフォールバックする。"""
            img = Image.open(crop_path).convert("RGB")
            img_arr = np.array(img).astype(np.float64)
            pixels = None
            if mask_path and os.path.exists(mask_path) and box is not None:
                x1, y1, x2, y2 = [int(round(v)) for v in box]
                mask_full = Image.open(mask_path).convert("L")
                mask_crop = mask_full.crop((x1, y1, x2, y2))
                mask_arr = np.array(mask_crop) > 127
                if mask_arr.shape[:2] == img_arr.shape[:2] and mask_arr.any():
                    pixels = img_arr[mask_arr]
            if pixels is None or len(pixels) == 0:
                pixels = img_arr.reshape(-1, 3)

            k = min(n_clusters, max(1, len(pixels) // 30))
            if k <= 1:
                return [(np.median(pixels, axis=0), 1.0)]

            rng = np.random.default_rng(0)
            sample = pixels if len(pixels) <= 5000 else pixels[rng.choice(len(pixels), 5000, replace=False)]
            centroids01, labels = kmeans2(sample / 255.0, k, seed=0, minit="++")
            centroids = np.clip(centroids01 * 255.0, 0, 255)
            counts = np.bincount(labels, minlength=k)
            weights = counts / max(1, counts.sum())
            order = np.argsort(-weights)
            return [(centroids[i], float(weights[i])) for i in order if weights[i] > 0]


        def _match_vertex_colors_to_targets(orig_colors_u8, target_clusters):
            """元メッシュの頂点色(N,4 uint8)を、写真側と同じ数までクラスタリングし、明度
            (Value)の順序で写真側のクラスタと対応付ける。各頂点について、対応する写真の
            目標色(RGB 0-255)を求め、(N,3)の配列として返す(頂点ごとに違う目標色を持てる
            ようにするため。例: 暗い脚の頂点は写真の暗い色へ、明るい座面の頂点は写真の
            明るい色へ、それぞれ別々に近づく)。"""
            colors01 = orig_colors_u8[:, :3].astype(np.float64) / 255.0
            n = len(colors01)
            target_rgbs = np.array([c for c, _w in target_clusters])
            k = min(len(target_clusters), max(1, n // 30))

            if k <= 1 or n < 30:
                return np.tile(target_rgbs[0], (n, 1))

            sample_n = min(n, 4000)
            rng = np.random.default_rng(0)
            sample_idx = rng.choice(n, sample_n, replace=False) if n > sample_n else np.arange(n)
            mesh_centroids, _ = kmeans2(colors01[sample_idx], k, seed=0, minit="++")

            # 全頂点を、最も近いメッシュ側クラスタ中心に割り当てる
            dists = np.linalg.norm(colors01[:, None, :] - mesh_centroids[None, :, :], axis=2)
            vertex_cluster = np.argmin(dists, axis=1)

            # メッシュ側クラスタ・写真側クラスタをそれぞれ明度(Value)順に並べて対応付ける
            mesh_v = mcolors.rgb_to_hsv(np.clip(mesh_centroids, 0.0, 1.0))[:, 2]
            mesh_order = np.argsort(mesh_v)
            target_v = mcolors.rgb_to_hsv(np.clip(target_rgbs / 255.0, 0.0, 1.0))[:, 2]
            target_order = np.argsort(target_v)

            cluster_to_target = np.zeros(k, dtype=int)
            for rank, mesh_ci in enumerate(mesh_order):
                t_rank = int(round(rank / max(1, k - 1) * (len(target_order) - 1))) if k > 1 else 0
                cluster_to_target[mesh_ci] = target_order[t_rank]

            return target_rgbs[cluster_to_target[vertex_cluster]]


        def _hue_shift_vertex_colors(vertex_colors_rgba_u8, target_rgb_per_vertex, hue_strength, saturation_strength,
                                      min_target_saturation=RECOLOR_MIN_TARGET_SATURATION):
            """頂点色の配列(N,4 uint8)に対し、色相(Hue)を頂点ごとの目標色
            (target_rgb_per_vertex: (N,3) or (3,)、0-255)の色相へhue_strengthだけ、彩度も
            saturation_strengthだけ寄せる。明度(Value)はそのまま保持する。写真側の彩度が
            min_target_saturation未満(白・黒・グレーに近い)の頂点は、色相を無理に合わせない
            (彩度が低い色は色相の値自体が数値的に不安定なため)。新しい頂点色配列を返す。"""
            colors01 = vertex_colors_rgba_u8.astype(np.float64) / 255.0
            rgb, alpha = colors01[:, :3], colors01[:, 3]

            hsv = mcolors.rgb_to_hsv(rgb)
            h, s, v = hsv[:, 0], hsv[:, 1], hsv[:, 2]

            target_rgb01 = np.clip(np.asarray(target_rgb_per_vertex, dtype=np.float64) / 255.0, 0.0, 1.0)
            if target_rgb01.ndim == 1:
                target_rgb01 = np.tile(target_rgb01, (len(rgb), 1))
            target_hsv = mcolors.rgb_to_hsv(target_rgb01)
            target_h, target_s = target_hsv[:, 0], target_hsv[:, 1]

            # 写真側の彩度が低いほど、色相を合わせる強さを弱める(無彩色に無理に色を足さない)
            effective_hue_strength = hue_strength * np.clip(target_s / max(min_target_saturation, 1e-6), 0.0, 1.0)

            # 色相は円環(0〜1が繋がっている)なので、最短経路で回す
            delta_h = (target_h - h + 0.5) % 1.0 - 0.5
            new_h = (h + delta_h * effective_hue_strength) % 1.0
            new_s = np.clip(s + (target_s - s) * saturation_strength, 0.0, 1.0)

            new_rgb = mcolors.hsv_to_rgb(np.stack([new_h, new_s, v], axis=1))
            new_colors = np.clip(new_rgb * 255.0, 0, 255).astype(np.uint8)
            new_alpha = np.clip(alpha * 255.0, 0, 255).astype(np.uint8)
            return np.column_stack([new_colors, new_alpha])

        # --- 前回このセルを実行した際にダウンロードした配置用モデルを削除してから始める
        #     (実行のたびにディスク使用量が増え続けないようにするため) ---
        if os.path.isdir(PLACEMENT_DOWNLOAD_DIR):
            shutil.rmtree(PLACEMENT_DOWNLOAD_DIR, ignore_errors=True)
        os.makedirs(PLACEMENT_DOWNLOAD_DIR, exist_ok=True)


        def _download_hssd_objects_to(uids, local_dir, max_workers=PLACEMENT_DOWNLOAD_WORKERS):
            """Section 12の_download_hssd_objects()と同じロジックだが、専用ディレクトリに
            ダウンロードする(処理後にまとめて削除しやすくするため)。"""
            uid_to_local_path = {}

            def _dl(uid):
                relpath = _uid_to_relpath_map.get(uid)
                if not relpath:
                    return uid, None
                try:
                    local_path = hf_hub_download(
                        repo_id=HSSD_REPO_ID, repo_type="dataset", filename=relpath,
                        local_dir=local_dir, token=HF_TOKEN,
                    )
                    return uid, local_path
                except Exception as e:
                    print(f"  [WARN] ダウンロード失敗: {uid}: {e!r}")
                    return uid, None

            with ThreadPoolExecutor(max_workers=max_workers) as pool:
                futures = [pool.submit(_dl, uid) for uid in uids]
                for future in as_completed(futures):
                    uid, local_path = future.result()
                    if local_path:
                        uid_to_local_path[uid] = local_path
            return uid_to_local_path


        # --- カメラの俯角・傾きにより、これまでは家具全体が斜めに配置されていた。
        #     ここでは「ワールドの上向き」を2通りの独立した手法で推定し、組み合わせることで
        #     より頑健に求める。
        #       手法A: MoGe-2の点群にRANSACで平面フィッティングし、床面の法線を検出する
        #               (単純な「画像下部×法線の平均」より、ラグの模様・家具の脚・法線推定
        #                ノイズなどの外れ値に強い)。
        #       手法B: WildDet3Dが検出済みの各家具OBBの「ローカルY軸(=高さ方向)」を集計する
        #               (多くの家具は床に直立して置かれているはず、という仮定に基づく。
        #                点群を使わないので、手法Aとは独立したクロスチェックになる)。
        #     両方求まった場合、角度差が小さければ(=互いに裏付け合っている)平均して使い、
        #     大きくズレていればインライア数が多い(=根拠が強い)手法Aを優先する。 ---
        _depth_npz_path = os.path.join(WORK_DIR, "depth", "moge_depth.npz")
        if os.path.isfile(_depth_npz_path):
            _floor_npz = np.load(_depth_npz_path)
            _floor_points_map = _floor_npz["points"]
            _floor_mask_map = _floor_npz["mask"].astype(bool)
        else:
            # ディスクに保存済みファイルが無い場合、Section 1.5のグローバル変数が
            # まだ残っていればそれを使う(同じカーネルでSection 1.5から続けて実行した場合)。
            _floor_points_map = points_map
            _floor_mask_map = depth_mask.astype(bool)

        _UP_GUESS_CV = np.array([0.0, -1.0, 0.0])  # OpenCVカメラ座標系(Y=下)での「仮の上向き」

        # --- Section 6で保存しておいた「床」のSAM3セグメンテーションマスクがあれば、それを
        #     RANSACの候補領域として使う(「画像下部◯%」という位置ヒューリスティックより、
        #     実際に床である画素だけに絞り込めるため精度が上がる)。無ければ、そのヒューリ
        #     スティックにフォールバックする。 ---
        _floor_seg_mask_path = os.path.join(WORK_DIR, "floor_mask.npy")
        _floor_seg_mask = None
        if os.path.isfile(_floor_seg_mask_path):
            _floor_seg_mask = np.load(_floor_seg_mask_path).astype(bool)
            if _floor_seg_mask.shape != _floor_mask_map.shape:
                print(f"  [WARN] floor_mask.npyの形状({_floor_seg_mask.shape})が"
                      f"深度マップ({_floor_mask_map.shape})と一致しないため無視します。")
                _floor_seg_mask = None
        elif ("floor_mask_2d" in locals() or "floor_mask_2d" in globals()) and floor_mask_2d is not None:
            # ディスクに保存済みファイルが無い場合、Section 6のグローバル変数が
            # まだ残っていればそれを使う
            _floor_seg_mask = np.asarray(floor_mask_2d).astype(bool)


        def _bottom_row_candidate_mask(mask, bottom_row_ratio=0.65):
            """SAM3の床マスクが無い場合のフォールバック: 画像下部(既定65%)の有効画素を
            候補領域とする、位置だけに基づいたヒューリスティック。"""
            h, w = mask.shape[:2]
            row_idx = np.arange(h)[:, None].repeat(w, axis=1)
            return mask & (row_idx >= int(h * (1.0 - bottom_row_ratio)))


        if _floor_seg_mask is not None:
            _floor_candidate_mask = _floor_mask_map & _floor_seg_mask
            _floor_candidate_source = f"SAM3の床セグメンテーション(画素数={int(_floor_candidate_mask.sum())})"
            if _floor_candidate_mask.sum() < 200:
                # SAM3マスクは検出できたが有効点が少なすぎる場合は、位置ヒューリスティックへ
                # フォールバックする
                _floor_candidate_mask = _bottom_row_candidate_mask(_floor_mask_map)
                _floor_candidate_source = "画像下部ヒューリスティック(SAM3マスクの有効点が不足)"
        else:
            _floor_candidate_mask = _bottom_row_candidate_mask(_floor_mask_map)
            _floor_candidate_source = "画像下部ヒューリスティック(SAM3の床マスクが見つからないため)"

        print(f"床のRANSAC候補領域: {_floor_candidate_source}")


        def _estimate_floor_normal_ransac(points_map, candidate_mask, up_guess, num_iterations=1000,
                                            distance_threshold=0.02, max_plane_tilt_deg=35.0, rng_seed=0):
            """candidate_mask(あらかじめ床候補として絞り込んだ画素のブール配列)内の点群に
            RANSACで平面フィッティングし、床面の法線を頑健に推定する。
            ((floor_normal_cv, inlier_points_cv)を返す。判定不能なら(None, None))。

            候補点から毎回3点をランダムに選んで平面を作る。up_guessから
            max_plane_tilt_deg度以上傾いた平面(壁など)は最初から候補から外し、
            インライア数(平面までの距離がdistance_threshold未満の点数)が最大の平面を
            採用したうえで、そのインライア点にSVDで最終的な法線を当てはめ直す。
            """
            pts = points_map[candidate_mask]
            n = len(pts)
            if n < 200:
                return None, None

            rng = np.random.default_rng(rng_seed)
            max_sample = min(n, 20000)
            if n > max_sample:
                pts = pts[rng.choice(n, size=max_sample, replace=False)]
            cos_tilt_max = np.cos(np.radians(max_plane_tilt_deg))

            best_inlier_mask, best_count = None, 0
            for _ in range(num_iterations):
                idx3 = rng.choice(len(pts), size=3, replace=False)
                p0, p1, p2 = pts[idx3]
                normal = np.cross(p1 - p0, p2 - p0)
                norm = np.linalg.norm(normal)
                if norm < 1e-8:
                    continue
                normal = normal / norm
                if np.dot(normal, up_guess) < 0:
                    normal = -normal
                if np.dot(normal, up_guess) < cos_tilt_max:
                    continue  # 壁など、水平から大きく傾いた平面は候補から除外
                d = -np.dot(normal, p0)
                dist = np.abs(pts @ normal + d)
                inlier_mask = dist < distance_threshold
                count = int(inlier_mask.sum())
                if count > best_count:
                    best_count = count
                    best_inlier_mask = inlier_mask

            if best_inlier_mask is None or best_count < 200:
                return None, None

            inlier_pts = pts[best_inlier_mask]
            centroid = inlier_pts.mean(axis=0)
            _, _, vt = np.linalg.svd(inlier_pts - centroid)
            floor_normal = vt[-1]
            floor_normal = floor_normal / (np.linalg.norm(floor_normal) + 1e-8)
            if np.dot(floor_normal, up_guess) < 0:
                floor_normal = -floor_normal
            return floor_normal, inlier_pts


        def _estimate_up_from_obb_axes(search_results, up_guess, max_tilt_deg=40.0):
            """WildDet3Dが検出した各家具OBBの「ローカルY軸(=高さ方向)」を集計し、その平均
            方向を推定上向きとして返す((up_normal_cv, 使用件数)。判定不能なら(None, 0))。

            多くの家具は床の上に直立して置かれている、という仮定に基づく。up_guessから
            max_tilt_deg度以上傾いている(＝直立していなさそうな、誤検出の可能性が高い)
            OBBは外れ値として除外する。
            """
            up_vectors = []
            for res_entry in search_results:
                obb = res_entry.get("obb_3d")
                if obb is None:
                    continue
                R_cv = trimesh.transformations.quaternion_matrix(
                    np.asarray(obb["quaternion_wxyz"], dtype=np.float64)
                )[:3, :3]
                local_up = R_cv[:, 1]  # WildDet3Dの規約: ローカルY軸=高さ方向
                local_up = local_up / (np.linalg.norm(local_up) + 1e-8)
                if np.dot(local_up, up_guess) < 0:
                    local_up = -local_up
                if np.dot(local_up, up_guess) >= np.cos(np.radians(max_tilt_deg)):
                    up_vectors.append(local_up)

            if len(up_vectors) < 3:
                return None, 0

            up_normal = np.mean(up_vectors, axis=0)
            up_normal = up_normal / (np.linalg.norm(up_normal) + 1e-8)
            return up_normal, len(up_vectors)


        _floor_normal_ransac, _floor_ransac_inlier_pts = _estimate_floor_normal_ransac(
            _floor_points_map, _floor_candidate_mask, _UP_GUESS_CV
        )
        _floor_ransac_inliers = 0 if _floor_ransac_inlier_pts is None else len(_floor_ransac_inlier_pts)
        _up_from_obb, _obb_up_count = _estimate_up_from_obb_axes(search_results, _UP_GUESS_CV)

        _final_up_cv = None
        if _floor_normal_ransac is not None and _up_from_obb is not None:
            agreement_deg = float(np.degrees(np.arccos(
                np.clip(np.dot(_floor_normal_ransac, _up_from_obb), -1.0, 1.0)
            )))
            print(f"床面法線(点群RANSAC, インライア{_floor_ransac_inliers}点) と "
                  f"家具OBBの直立方向(集計{_obb_up_count}件) の角度差: {agreement_deg:.1f}度")
            if agreement_deg <= 20.0:
                # 両手法がおおむね一致 → 平均して使う(お互いの推定を裏付け合っているので、
                # より頑健な推定になる)
                _final_up_cv = _floor_normal_ransac + _up_from_obb
                _final_up_cv = _final_up_cv / (np.linalg.norm(_final_up_cv) + 1e-8)
                print("  -> 両手法を平均して採用します。")
            else:
                # 大きくズレている場合は、インライア数が多い(＝根拠が強い)点群RANSACを優先する
                _final_up_cv = _floor_normal_ransac
                print("  -> ズレが大きいため、点群RANSACによる床法線を優先します。")
        elif _floor_normal_ransac is not None:
            _final_up_cv = _floor_normal_ransac
            print(f"床面法線を点群RANSACのみで推定しました(インライア{_floor_ransac_inliers}点)。")
        elif _up_from_obb is not None:
            _final_up_cv = _up_from_obb
            print(f"点群からの床平面検出に失敗したため、家具OBBの直立方向のみで推定しました"
                  f"(集計{_obb_up_count}件)。")
        else:
            print("床面・家具OBBのどちらからもワールドの上向きを推定できなかったため、"
                  "整列を行わずカメラ座標系のまま配置します。")

        if _final_up_cv is not None:
            # 推定した上向き(_final_up_cv)を「仮の上向き」(_UP_GUESS_CV)へ回転させる行列を、
            # そのままワールド整列の回転として使う(以後、床は水平になる)。
            _align_transform_4x4 = trimesh.geometry.align_vectors(_final_up_cv, _UP_GUESS_CV)
            _R_align = _align_transform_4x4[:3, :3]

            if _floor_ransac_inlier_pts is not None:
                # 実際に検出できた床のインライア点があれば、それを整列後のY=0基準にする
                _floor_points_aligned = (_R_align @ _floor_ransac_inlier_pts.T).T
                _floor_y_world = float(np.median(_floor_points_aligned[:, 1]))
            else:
                # 床平面自体は検出できず、家具OBBの直立方向だけで整列した場合は、
                # 有効点群全体の下位(カメラ座標のYが大きい=下側)側から大まかに床の高さを見積もる
                _all_points_aligned = (_R_align @ _floor_points_map[_floor_mask_map].T).T
                _floor_y_world = float(np.percentile(_all_points_aligned[:, 1], 90))

            tilt_deg = float(np.degrees(np.arccos(np.clip(np.dot(_final_up_cv, _UP_GUESS_CV), -1.0, 1.0))))
            print(f"ワールド整列を適用しました(元の上向きからの補正角: {tilt_deg:.1f}度)。")
        else:
            _R_align = np.eye(3)
            _floor_y_world = 0.0

        print("向きの自動推定(レンダリング+CLIP照合)は現在も無効化されており、"
              "床の水平整列以外はWildDet3Dの推論した回転をそのまま使います。")

        # WildDet3D(vis4d)はOpenCV系のカメラ座標(X=右, Y=下, Z=カメラから奥へ)で
        # center_xyz・回転を出力していると考えられるが、glTF/model-viewerはY-up・
        # 奥行きが-Z(カメラから見て奥が-Z方向)という別の規約を使う。OpenCV -> glTF の変換は
        # 「Y軸とZ軸の符号を反転する」(X軸まわりに180度回す、行列式+1の正しい回転)。
        # 位置ベクトルは単純にMを左から掛けるだけでよいが、回転行列(姿勢)は基底変換になるので
        # M @ R @ M^-1 (=M @ R @ M, Mは対称直交行列なのでM^-1=M)という「共役」の形で変換する
        # 必要がある(単純にM @ Rとするのは誤り)。
        _M_cv_to_gl = np.diag([1.0, -1.0, -1.0])


        def _to_gltf_pose(center_xyz, quaternion_wxyz):
            """WildDet3D由来の (center_xyz, quaternion_wxyz) を、ワールド整列(_R_align、
            現状は恒等変換)を適用したうえで、glTFの座標系(Y-up)での (位置, 3x3回転行列) に変換する。"""
            center_cv = np.asarray(center_xyz, dtype=np.float64)
            R_cv = trimesh.transformations.quaternion_matrix(
                np.asarray(quaternion_wxyz, dtype=np.float64)
            )[:3, :3]

            aligned_position = _R_align @ center_cv
            aligned_position[1] -= _floor_y_world
            aligned_rotation = _R_align @ R_cv

            world_position = _M_cv_to_gl @ aligned_position
            world_rotation = _M_cv_to_gl @ aligned_rotation @ _M_cv_to_gl
            return world_position, world_rotation


        def _best_shape_fit_deg(extents, target_size_whl):
            """候補メッシュ(extents)の自然な長辺(ローカルX/Zのどちらが長いか)を、目標
            (target_size_whl=[幅w,高さh,長さl])のlength/widthのうち大きい方に合わせるために
            必要な追加ヨー回転(0 または 90)を返す。候補モデル自身の縦横比をできるだけ保つ
            ための選択で、正面の向きを気にしないオブジェクト(椅子以外)に使う。"""
            ext_x = max(extents[0], 1e-6)
            ext_z = max(extents[2], 1e-6)
            target = np.asarray(target_size_whl, dtype=np.float64)
            target_width, target_length = target[0], target[2]
            cand_x_is_longer = ext_x >= ext_z
            target_length_is_longer = target_length >= target_width
            return 0 if cand_x_is_longer == target_length_is_longer else 90


        def _scale_for_extra_rotation(extents, target_size_whl, extra_deg):
            """候補メッシュ(extents)に対し、「world_rotationに加えてextra_deg度(0/90/180/270)
            だけ追加でヨー回転させる」ことを前提に、target_size_whl([幅w,高さh,長さl])へ
            一致させるための軸ごとのスケールを計算する。

            注意: WildDet3Dの回転行列は「ローカルX軸=長さ(l)、Z軸=幅(w)」という規約
            (Section 7.6の_obb_corners_camで確認済み)。extra_degが90度単位で奇数倍
            (90または270)の場合、ローカルX軸は追加回転後にworldの幅方向、ローカルZ軸は
            長さ方向に写像されるため、割り当てを入れ替える。

            OBB(size_whl・world_position・world_rotation)自体はここでは一切変更しない。
            「回転をどう選ぶか」を先に決めてから、その回転に矛盾しないスケールを
            後から計算する、という順番になっている点がポイント(逆順にすると、正面の
            向きの探索がスケール都合に制約されてしまい、正しい正面が選べなくなることが
            あった)。
            """
            ext_x = max(extents[0], 1e-6)
            ext_z = max(extents[2], 1e-6)
            target = np.asarray(target_size_whl, dtype=np.float64)
            target_width, target_height, target_length = target[0], target[1], target[2]

            scale = np.ones(3, dtype=np.float64)
            scale[1] = target_height / max(extents[1], 1e-6)

            if extra_deg % 180 == 0:
                scale[0] = target_length / ext_x  # ローカルX軸 = 長さ(l)
                scale[2] = target_width / ext_z   # ローカルZ軸 = 幅(w)
            else:
                scale[0] = target_width / ext_x
                scale[2] = target_length / ext_z

            return scale


        def _build_placement_transform(local_center, scale_xyz, rotation_matrix_3x3, world_position):
            """「原点へ平行移動→スケール→回転→ワールド座標での最終位置へ平行移動」の
            4x4変換行列を組み立てる。"""
            T_center = np.eye(4)
            T_center[:3, 3] = -np.asarray(local_center, dtype=np.float64)

            S = np.eye(4)
            S[0, 0], S[1, 1], S[2, 2] = scale_xyz

            R = np.eye(4)
            R[:3, :3] = rotation_matrix_3x3

            T_place = np.eye(4)
            T_place[:3, 3] = np.asarray(world_position, dtype=np.float64)

            return T_place @ R @ S @ T_center


        def _make_obb_box_mesh(center_xyz, size_whl, quaternion_wxyz, color_rgba=(255, 60, 0, 90)):
            """OBB(中心・サイズ・姿勢)から、半透明の色付きボックスメッシュを作る
            (問題切り分け用のデバッグ可視化。実際に配置したモデルの形・位置とOBBがどれだけ
            ズレているかを目視確認できるようにする)。

            注意: WildDet3Dの回転行列は「ローカルX軸=長さ(l), Y軸=高さ(h), Z軸=幅(w)」という
            規約(Section 7.6の_obb_corners_camで確認済み)。trimesh.creation.boxのextentsも
            この順(l, h, w)で渡す必要がある。
            """
            w, h, l = size_whl
            box = trimesh.creation.box(extents=[l, h, w])  # ローカルX=長さ,Y=高さ,Z=幅(WildDet3Dの規約に合わせる)
            world_position, world_rotation = _to_gltf_pose(center_xyz, quaternion_wxyz)
            T = np.eye(4)
            T[:3, 3] = world_position
            R = np.eye(4)
            R[:3, :3] = world_rotation
            box.apply_transform(T @ R)

            material = trimesh.visual.material.PBRMaterial(
                baseColorFactor=color_rgba, alphaMode="BLEND", doubleSided=True,
            )
            box.visual = trimesh.visual.TextureVisuals(material=material)
            return box


        def _make_front_arrow_mesh(world_position, direction_gl, length=0.5, color_rgba=(0, 220, 0, 255)):
            """world_position から direction_gl 方向に伸びる矢印メッシュを作る
            (デバッグ用。候補の3Dモデル自体の「正面」の目安がどちらを向いているかを可視化する)。

            「正面」の軸は、cand_lo_axis(目標サイズのwidth/lengthのうち小さい方を割り当てた
            ローカル軸)を使う(実際にどちらの符号(+/-)が正面かまでは分からないため、
            ここでは+方向を仮の正面として描画している)。
            """
            shaft_radius = length * 0.03
            head_radius = length * 0.08
            head_length = length * 0.25
            shaft_length = length - head_length

            shaft = trimesh.creation.cylinder(radius=shaft_radius, height=shaft_length, sections=8)
            shaft.apply_translation([0.0, 0.0, shaft_length / 2.0])  # 底面がZ=0に来るようにする
            head = trimesh.creation.cone(radius=head_radius, height=head_length, sections=8)
            head.apply_translation([0.0, 0.0, shaft_length])  # 底面がシャフトの先端(Z=shaft_length)に来るようにする
            arrow = trimesh.util.concatenate([shaft, head])

            # 矢印はデフォルトで+Z方向を向いているので、それをdirection_glへ向ける回転を求める
            direction_gl = np.asarray(direction_gl, dtype=np.float64)
            direction_gl = direction_gl / (np.linalg.norm(direction_gl) + 1e-8)
            align_transform = trimesh.geometry.align_vectors(np.array([0.0, 0.0, 1.0]), direction_gl)

            T = np.eye(4)
            T[:3, 3] = np.asarray(world_position, dtype=np.float64)
            arrow.apply_transform(T @ align_transform)

            material = trimesh.visual.material.PBRMaterial(baseColorFactor=color_rgba)
            arrow.visual = trimesh.visual.TextureVisuals(material=material)
            return arrow


        def _yaw_matrix(degrees):
            """ワールド(glTF, Y-up)のY軸まわりに degrees 度回転する3x3回転行列を返す。
            椅子系オブジェクトの「正面」候補を90度刻みで生成するために使う
            (Y軸まわりの回転どうしは可換なので、ローカル座標系で回転させても
            world_rotationを左から掛けた後の向きの候補として扱える)。
            """
            theta = np.radians(degrees)
            c, s = np.cos(theta), np.sin(theta)
            return np.array([
                [c, 0.0, s],
                [0.0, 1.0, 0.0],
                [-s, 0.0, c],
            ])


        def _make_upright_rotation(rotation_matrix_3x3, world_up=np.array([0.0, 1.0, 0.0])):
            """回転行列(3x3)を、ローカルY軸(WildDet3Dの規約: 高さ方向)がワールドの上向きに
            正確に一致するように補正する(＝床面に整列させた後、個々の家具も真っ直ぐ
            (ピッチ・ロールが無い状態)に立つように角度を調整する)。

            WildDet3Dは家具ごとに独立して姿勢を推定しているため、シーン全体を床に合わせて
            整列(_R_align)した後も、個々の家具の回転には推定誤差による微小な傾き
            (ピッチ・ロール)が残ることがある。ここでは、鉛直軸まわりの向き(ヨー)は
            できるだけ維持したまま、それ以外の傾きだけを取り除く:
            元のローカルX軸(進行方向の目安)を水平面(ワールドの上向きに垂直な面)へ投影して
            新しいX軸とし、Y軸をワールドの上向きに固定、Z軸はそれらの外積から作り直して
            右手系の正規直交基底に組み直す。
            ローカルX軸がほぼ真上/真下を向いていて水平成分が取れない場合は、ローカルZ軸を
            代わりに基準として使う。
            """
            x_axis = rotation_matrix_3x3[:, 0]
            x_horiz = x_axis - np.dot(x_axis, world_up) * world_up
            if np.linalg.norm(x_horiz) >= 1e-6:
                x_new = x_horiz / np.linalg.norm(x_horiz)
                z_new = np.cross(x_new, world_up)
                z_new = z_new / (np.linalg.norm(z_new) + 1e-8)
                return np.column_stack([x_new, world_up, z_new])

            z_axis = rotation_matrix_3x3[:, 2]
            z_horiz = z_axis - np.dot(z_axis, world_up) * world_up
            if np.linalg.norm(z_horiz) < 1e-6:
                return rotation_matrix_3x3  # どうしても水平の基準が取れない場合は補正しない

            z_new = z_horiz / np.linalg.norm(z_horiz)
            x_new = np.cross(world_up, z_new)
            x_new = x_new / (np.linalg.norm(x_new) + 1e-8)
            return np.column_stack([x_new, world_up, z_new])


        # ==========================================
        CHAIR_KEYWORDS = [
            "chair", "sofa", "loveseat", "recliner", "chaise", "bean bag",
            "stool", "ottoman", "bench",
        ]  # 「椅子系」とみなすカテゴリ名のキーワード(ソファ・ベンチなども座る向きがあるため含める)
        MIN_BACKREST_ASYMMETRY = 0.15  # 前後の高さの差が、全体の高さに対してこの比率以上あれば
                                         # 「背もたれがある」と判定する(小さすぎる差はノイズとみなす)
        TABLE_KEYWORDS = ["table", "desk"]  # 「テーブル系」とみなすカテゴリ名のキーワード
                                              # (椅子系オブジェクトを向かせる向き先の候補として使う)
        # ==========================================


        def _is_chair_like(category):
            cat_lower = category.lower()
            return any(kw in cat_lower for kw in CHAIR_KEYWORDS)


        def _is_table_like(category):
            cat_lower = category.lower()
            return any(kw in cat_lower for kw in TABLE_KEYWORDS)


        def _estimate_forward_by_backrest_asymmetry(mesh_centered_raw, min_asymmetry=MIN_BACKREST_ASYMMETRY):
            """椅子の多くは「背もたれ側は背が高く、座面側(手前)は低い」という高さの非対称性を
            持つことを利用して、実際の前後軸・向きを推定する。

            ローカルX軸・Z軸それぞれについて、原点を境に+側/-側に分け、それぞれの最大の高さ
            (Y方向)を比較する。差が大きい方の軸を前後軸とみなし、高さが低い方(＝座面側)を
            前方向とする。どちらの軸でも十分な非対称性が見られない(背もたれの無い丸椅子等)場合は
            Noneを返す(＝寸法ベースのcand_lo_axisにフォールバックする合図)。
            """
            verts = mesh_centered_raw.vertices
            bounds = mesh_centered_raw.bounds
            y_min = bounds[0][1]
            total_height = max(bounds[1][1] - bounds[0][1], 1e-6)

            best_axis, best_sign, best_asym = None, None, 0.0
            for axis in (0, 2):
                pos_mask = verts[:, axis] >= 0
                neg_mask = ~pos_mask
                if pos_mask.sum() < 10 or neg_mask.sum() < 10:
                    continue
                pos_height = verts[pos_mask, 1].max() - y_min
                neg_height = verts[neg_mask, 1].max() - y_min
                asym = abs(pos_height - neg_height) / total_height
                if asym > best_asym:
                    best_asym = asym
                    best_axis = axis
                    # 高さが低い方(座面側)を前方向とする
                    best_sign = -1.0 if pos_height > neg_height else 1.0

            if best_axis is None or best_asym < min_asymmetry:
                return None

            forward = np.zeros(3)
            forward[best_axis] = best_sign
            return forward


        MIN_CENTROID_OFFSET_RATIO = 0.05  # 重心のズレが、水平方向の広がりに対してこの比率未満なら
                                            # 「判定不能」とみなす


        def _estimate_forward_by_height_weighted_centroid(mesh_centered_raw, min_offset_ratio=MIN_CENTROID_OFFSET_RATIO):
            """_estimate_forward_by_backrest_asymmetry()のフォールバック。

            頂点の「高さ(Y座標)で重み付けした重心」と、通常の(重み無し)重心を、XZ平面上で
            比較する。高い部分(背もたれ)に重み付き重心が引っ張られるので、そのズレと逆方向を
            「正面」とみなす。ローカルX軸・Z軸に限定されないため、斜めに作られた背もたれにも
            対応できる(その代わり、二値分割ほどはっきりした閾値を引けないので、ズレが小さすぎる
            場合は判定不能としてNoneを返す)。
            """
            verts = mesh_centered_raw.vertices
            bounds = mesh_centered_raw.bounds
            y_min = bounds[0][1]
            heights = verts[:, 1] - y_min
            total_height = float(heights.sum())
            if total_height <= 1e-6:
                return None

            weighted_centroid_xz = (verts[:, [0, 2]] * heights[:, None]).sum(axis=0) / total_height
            plain_centroid_xz = verts[:, [0, 2]].mean(axis=0)
            offset_xz = weighted_centroid_xz - plain_centroid_xz

            # 正規化用に、水平方向の広がり(対角線の半分程度)を基準スケールとする
            horiz_extent = bounds[1][[0, 2]] - bounds[0][[0, 2]]
            scale = float(np.linalg.norm(horiz_extent)) / 2.0
            if scale <= 1e-6:
                return None

            offset_norm = float(np.linalg.norm(offset_xz))
            if offset_norm / scale < min_offset_ratio:
                return None  # ズレが小さすぎて判定できない

            # 連続的な方向ではなく、一番近い元の家具のローカル軸(+X/-X/+Z/-Z)にスナップする
            # (_estimate_forward_by_backrest_asymmetry()と同じ、軸に沿った形式で返すことで、
            #  後段の処理(矢印の描画、向きに関するルールなど)と扱いを揃える)
            if abs(offset_xz[0]) >= abs(offset_xz[1]):
                axis, sign = 0, (-1.0 if offset_xz[0] > 0 else 1.0)
            else:
                axis, sign = 2, (-1.0 if offset_xz[1] > 0 else 1.0)

            forward = np.zeros(3)
            forward[axis] = sign
            return forward


        def _pick_forward_facing_nearest_table(base_local_forward, world_rotation, world_position,
                                                table_world_positions, candidate_degrees):
            """base_local_forward(背もたれの非対称性/重心のズレから求めた「本当の正面」の
            ローカルベクトル)を基準に、それをY軸まわりに candidate_degrees 度だけ回転させた
            各候補(通常は[0, 90, 180, 270]の4パターン)を作り、world_rotationを適用して
            ワールド空間での向きを求める。

            その中から、このオブジェクトの位置(world_position)から見て最も近いテーブル系家具
            (table_world_positions)がある方向に、水平面(XZ平面)上で最も近い(内積が最大の)
            候補を選んで返す。テーブル系家具が1件も無い場合は、回転補正なし(候補の先頭、
            通常は0度 = base_local_forwardそのまま)を返す。

            戻り値: (選ばれた回転角度[度], 回転補正後のローカル正面ベクトル)
            """
            if not table_world_positions:
                return candidate_degrees[0], base_local_forward

            obj_xz = np.array([world_position[0], world_position[2]])
            nearest_table_xz = min(
                (np.array([p[0], p[2]]) for p in table_world_positions),
                key=lambda p: np.linalg.norm(p - obj_xz),
            )
            to_table_xz = nearest_table_xz - obj_xz
            to_table_norm = np.linalg.norm(to_table_xz)
            if to_table_norm < 1e-6:
                return candidate_degrees[0], base_local_forward
            to_table_xz = to_table_xz / to_table_norm

            best_deg = candidate_degrees[0]
            best_local_forward = base_local_forward
            best_score = -np.inf
            for deg in candidate_degrees:
                local_forward = _yaw_matrix(deg) @ base_local_forward
                world_dir = world_rotation @ local_forward
                world_dir_xz = np.array([world_dir[0], world_dir[2]])
                world_dir_norm = np.linalg.norm(world_dir_xz)
                if world_dir_norm < 1e-6:
                    continue
                world_dir_xz = world_dir_xz / world_dir_norm
                score = float(np.dot(world_dir_xz, to_table_xz))
                if score > best_score:
                    best_score = score
                    best_deg = deg
                    best_local_forward = local_forward

            return best_deg, best_local_forward


        # --- 各検出について、検索1位候補のUIDとOBB(位置・姿勢・サイズ)を集める ---
        _placement_entries = []
        for entry in search_results:
            if not entry["candidates"]:
                continue
            obb = entry.get("obb_3d")
            if obb is None:
                print(f"  [WARN] OBBが無いためスキップ: {entry['file']}")
                continue
            top1 = entry["candidates"][0]
            _placement_entries.append({
                "uid": top1["name"],
                "center_xyz": obb["center_xyz"],
                "size_whl": obb["size_whl"],
                "quaternion_wxyz": obb["quaternion_wxyz"],
                "category": entry["category"],
                "file": entry.get("file"),            # 色合わせ用: 検出時のクロップ画像
                "box": entry.get("box"),               # 色合わせ用: マスクをクロップ範囲に合わせるため
                "mask_file": entry.get("mask_file"),   # 色合わせ用: SAM3マスク
            })

        print(f"配置対象: {len(_placement_entries)} / {len(search_results)}件")

        # --- 椅子系オブジェクトを「最寄りのテーブル系家具の方向」へ向かせるために、
        #     全エントリのワールド位置・回転をあらかじめ計算しておく(center_xyz・
        #     quaternion_wxyzだけで求まるので、メッシュのダウンロード前でも計算できる)。
        #     床面をワールドのY軸に整列した後も、WildDet3Dは家具ごとに独立して姿勢を
        #     推定しているため、個々の家具の回転には推定誤差による微小な傾き(ピッチ・
        #     ロール)が残ることがある。ここで_make_upright_rotation()により、ヨー(鉛直軸
        #     まわりの向き)は維持したまま、全ての家具を床面に対して真っ直ぐ立たせる。
        #     あわせて、テーブル系家具のワールド位置一覧も作っておく。 ---
        for e in _placement_entries:
            _raw_world_position, _raw_world_rotation = _to_gltf_pose(e["center_xyz"], e["quaternion_wxyz"])
            e["world_position"] = _raw_world_position
            e["world_rotation"] = _make_upright_rotation(_raw_world_rotation)

        # --- OBBが床とほぼ面している(=床に接して置かれているはずの)家具は、位置を補正して
        #     正確に床面(ワールドY=0)の上に乗せる。upright化によりローカルY軸(高さ方向)は
        #     ワールドYと正確に一致しているので、OBB底面のワールドYは
        #     center_y - 高さ/2 で厳密に求まる(回転による投影を考える必要が無い)。
        #     壁掛け・天井吊りなど、そもそも床から離れている家具はこの補正の対象外とする
        #     (床とのギャップがFLOOR_SNAP_THRESHOLDを超える場合はそのままにする)。 ---
        FLOOR_SNAP_THRESHOLD = 0.08  # 家具底面と床(Y=0)との差がこの範囲(既定8cm)以内なら、
                                       # 「床に接している」とみなして正確にY=0へスナップする

        _n_snapped_to_floor = 0
        for e in _placement_entries:
            half_height = float(e["size_whl"][1]) / 2.0
            obb_bottom_y = float(e["world_position"][1]) - half_height
            if abs(obb_bottom_y) <= FLOOR_SNAP_THRESHOLD:
                e["world_position"] = e["world_position"].copy()
                e["world_position"][1] -= obb_bottom_y  # 底面がちょうどY=0に来るよう平行移動
                _n_snapped_to_floor += 1

        print(f"OBBが床とほぼ面していた家具: {_n_snapped_to_floor}/{len(_placement_entries)}件を"
              f"床面(Y=0)へ位置補正しました(閾値: 底面と床の差が{FLOOR_SNAP_THRESHOLD * 100:.0f}cm以内)。")

        _table_world_positions = [e["world_position"] for e in _placement_entries if _is_table_like(e["category"])]
        print(f"テーブル系家具として認識: {len(_table_world_positions)}件(椅子系オブジェクトの向き先探索に使用)")

        # --- 1位候補のGLBをまとめてダウンロード(専用フォルダへ。後で削除する) ---
        _placement_uids = sorted({e["uid"] for e in _placement_entries})
        _placement_uid_to_path = _download_hssd_objects_to(_placement_uids, PLACEMENT_DOWNLOAD_DIR)
        print(f"ダウンロード完了: {len(_placement_uid_to_path)} / {len(_placement_uids)}")

        _n_placed = 0
        for e in _placement_entries:
            local_path = _placement_uid_to_path.get(e["uid"])
            if not local_path:
                print(f"  [WARN] {e['uid']} のダウンロードに失敗しているためスキップ")
                continue
            try:
                mesh = trimesh.load(local_path, force="mesh")

                # --- 色合わせ: デシメーションで失われる前に、元の頂点位置・頂点色を保存しておく ---
                _orig_vertices_for_color = mesh.vertices.copy()
                _orig_vertex_colors_for_color = _bake_vertex_colors(mesh)

                if len(mesh.faces) > PLACEMENT_MAX_FACES:
                    v, f = _decimate_with_pyfqmr(
                        mesh.vertices, mesh.faces, PLACEMENT_MAX_FACES, PLACEMENT_DECIMATE_AGGRESSIVENESS
                    )
                    mesh = trimesh.Trimesh(vertices=v, faces=f, process=False)

                # --- 色合わせ: 新しい頂点それぞれに、最も近い元頂点の色を引き継がせたうえで、
                #     色相を写真の目標色へ寄せる ---
                if RECOLOR_TO_MATCH_PHOTO:
                    if "target_color_clusters" not in e:
                        _crop_path = os.path.join(output_dir, e["file"]) if e.get("file") else None
                        _mask_path = os.path.join(output_dir, e["mask_file"]) if e.get("mask_file") else None
                        if _crop_path and os.path.exists(_crop_path):
                            e["target_color_clusters"] = _extract_target_color_clusters(_crop_path, _mask_path, e.get("box"))
                        else:
                            e["target_color_clusters"] = None

                    if e.get("target_color_clusters"):
                        _tree = cKDTree(_orig_vertices_for_color)
                        _nn_dist, _nn_idx = _tree.query(mesh.vertices, k=1)
                        _inherited_colors = _orig_vertex_colors_for_color[_nn_idx]
                        _per_vertex_target_rgb = _match_vertex_colors_to_targets(
                            _inherited_colors, e["target_color_clusters"]
                        )
                        _new_colors = _hue_shift_vertex_colors(
                            _inherited_colors, _per_vertex_target_rgb,
                            RECOLOR_HUE_STRENGTH, RECOLOR_SATURATION_STRENGTH,
                        )
                        mesh.visual = trimesh.visual.color.ColorVisuals(mesh=mesh, vertex_colors=_new_colors)

                local_center = mesh.bounds.mean(axis=0)
                extents = mesh.bounds[1] - mesh.bounds[0]
                mesh_centered_raw = mesh.copy()
                mesh_centered_raw.apply_translation(-local_center)

                world_position, world_rotation = e["world_position"], e["world_rotation"]

                # cand_lo_axis: 「奥行き寄り」の目安(正面推定の暫定フォールバック・デバッグ矢印の
                # 向きに使用)。目標サイズ(size_whl)のwidth/lengthのどちらが小さいかで機械的に
                # 決める(OBB自体は変更しないので、回転の選び方とは無関係に決まる)。
                cand_lo_axis = 0 if e["size_whl"][2] <= e["size_whl"][0] else 2

                # --- 回転(何度追加でヨー回転させるか)を先に決めてから、その回転に合わせて
                #     スケールを後から計算する、という順番にしている。OBB(size_whl・
                #     world_position・world_rotation)自体は最初から最後まで一切変更せず、
                #     候補モデルをそのOBBの中にどの向きで置くか(4通り)だけを選ぶイメージ。
                #     (逆に「先にスケールの都合で回転を絞り込み、そのあと正面を探す」順序に
                #     すると、モデルの縦横比の都合で本来選ぶべき正面が候補から締め出されて
                #     しまうことがあった。) ---
                is_chair = _is_chair_like(e["category"])
                chair_local_forward = None

                if is_chair:
                    # 椅子系オブジェクトは、背もたれの高さの非対称性(ダメなら重心のズレ)から
                    # 計算した「本当の正面」ベクトルを使う。
                    chair_local_forward = _estimate_forward_by_backrest_asymmetry(mesh_centered_raw)
                    method_used = "背もたれの非対称性(軸ベース)"
                    if chair_local_forward is None:
                        # 軸ベースで判定できなかった場合、高さで重み付けした重心のズレを見る
                        # フォールバックを試す(斜めの背もたれなど、軸に沿わないケースに対応)
                        chair_local_forward = _estimate_forward_by_height_weighted_centroid(mesh_centered_raw)
                        method_used = "重心のズレ(軸非依存)"
                    if chair_local_forward is None:
                        # どちらの手法でも判定できない場合(丸椅子など、明確な非対称性が無い
                        # タイプ)は、ローカルX軸を仮の正面としておく(どちらでも良い。この後の
                        # 4方向探索で実際の向きが決まるので、大きな影響はない)。
                        chair_local_forward = np.array([1.0, 0.0, 0.0])
                        method_used = "正面不明(4方向探索に委ねる)"

                    # モデル自身の検出結果(chair_local_forward)を基準に、0/90/180/270度の
                    # 4通り全てを試し、最寄りのテーブル系家具の方向に最も近いものを選ぶ。
                    # (以前はここをshape_fit_degに合わせて2択に絞っていたが、shape_fit_degは
                    # スケール合わせのための基準で正面検出とは無関係のため、正しい正面候補が
                    # 締め出されてしまうことがあった。)
                    candidate_degrees = [0, 90, 180, 270]
                    chosen_deg, chair_local_forward = _pick_forward_facing_nearest_table(
                        chair_local_forward, world_rotation, world_position,
                        _table_world_positions, candidate_degrees,
                    )
                    print(f"    '{e['category']}': {method_used}で正面を計算 / "
                          f"最寄りのテーブル系家具の方向へ{chosen_deg}度回転補正")
                else:
                    # 椅子以外は正面の向きを気にしないので、候補モデル自身の縦横比を
                    # できるだけ保てる方の回転(0 または 90度)を選ぶ。
                    chosen_deg = _best_shape_fit_deg(extents, e["size_whl"])

                # 選んだ回転(chosen_deg)に矛盾しないよう、スケールをこの後に計算する。
                scale_xyz = _scale_for_extra_rotation(extents, e["size_whl"], chosen_deg)

                final_rotation = world_rotation @ _yaw_matrix(chosen_deg)
                # 最終回転(正面補正・形状補正込み)と、選んだ追加回転角度を_placement_entries
                # 側にも書き戻しておく。以後のセル(物理コライダー生成など)で、実際に配置
                # されたメッシュと同じ向き・同じスケールを再現できるようにするため。
                e["world_rotation"] = final_rotation
                e["mesh_extra_yaw_deg"] = chosen_deg

                _n_placed += 1
            except Exception as ex:
                print(f"  [WARN] 配置失敗: {e['uid']} ({e['category']}): {ex!r}")

        print(f"配置完了: {_n_placed} / {len(_placement_entries)}件")

        # --- ダウンロードした配置用モデルは、シーンへの埋め込みが終わったら削除する
        #     (glbファイル自体は既にシーンにbakeされているので、元ファイルは不要) ---
        shutil.rmtree(PLACEMENT_DOWNLOAD_DIR, ignore_errors=True)
        print(f"ダウンロード済みの配置用モデルを削除しました: {PLACEMENT_DOWNLOAD_DIR}")


        # ---- room_area (元ノートブック cell 48) ----
        # ==========================================
        OUTER_BOUNDARY_N_BINS = 360         # 360度を何分割して「その方向でいちばん遠い点」を探すか
        OUTER_BOUNDARY_PERCENTILE = 100.0   # 各方向で採用する点の遠さの分位点(100=真の最遠点。
                                             # ノイズで極端に遠い外れ値が混じる場合は98あたりに下げると
                                             # 安定する)
        MIN_BIN_POINTS = 3                  # その方向に何点以上あれば採用するか(少なすぎるとノイズに弱い)
        POINT_CLOUD_MAX_POINTS = 200000     # 表示・計算を軽くするための間引き上限
        # ==========================================

        # --- 深度点群を読み込む(Section 13.7と同じフォールバック方式) ---
        if ("points_map" in locals() or "points_map" in globals()) and ("depth_mask" in locals() or "depth_mask" in globals()):
            _rb_points_map = points_map
            _rb_mask_map = depth_mask.astype(bool)
        else:
            _depth_npz_path = os.path.join(WORK_DIR, "depth", "moge_depth.npz")
            if not os.path.isfile(_depth_npz_path):
                raise FileNotFoundError(
                    "深度点群が見つかりません。Section 1.5(MoGe-2深度推定)を先に実行してください。"
                )
            _rb_npz = np.load(_depth_npz_path)
            _rb_points_map = _rb_npz["points"]
            _rb_mask_map = _rb_npz["mask"].astype(bool)

        _rb_points_cv = _rb_points_map[_rb_mask_map]

        if len(_rb_points_cv) > POINT_CLOUD_MAX_POINTS:
            _rng = np.random.default_rng(0)
            _sel = _rng.choice(len(_rb_points_cv), POINT_CLOUD_MAX_POINTS, replace=False)
            _rb_points_cv = _rb_points_cv[_sel]

        # --- Section 12.5/13.7と同じ変換で、家具と同じワールド座標系(glTF, Y-up)に揃える ---
        _rb_R_align = globals().get("_R_align", np.eye(3))
        _rb_floor_y_world = globals().get("_floor_y_world", 0.0)
        _rb_M_cv_to_gl = globals().get("_M_cv_to_gl", np.diag([1.0, -1.0, -1.0]))

        _rb_points_aligned = (_rb_R_align @ _rb_points_cv.T).T
        _rb_points_aligned[:, 1] -= _rb_floor_y_world
        _rb_points_world = (_rb_M_cv_to_gl @ _rb_points_aligned.T).T

        # --- XZ平面(上から見た平面図)へ投影する ---
        _rb_xz = _rb_points_world[:, [0, 2]]

        # --- カメラ位置を中心に、放射状(角度ごと)にいちばん外側の点を選んでつなぐ。
        #     単眼深度推定の点群はカメラから四方八方へ放射状に広がっている形をしているため、
        #     「カメラを中心に、各方向でいちばん遠い点」を角度順につなぐだけで、部屋の外形に
        #     近い輪郭が作れる(Delaunay分割や輪郭のトレースが不要な、単純な方法)。
        #     このワールド座標系では、カメラの位置は変換の性質上ちょうど原点(0, 0)になる。 ---
        _rb_camera_xz = np.array([0.0, 0.0])

        _rb_rel = _rb_xz - _rb_camera_xz
        _rb_angles = np.arctan2(_rb_rel[:, 1], _rb_rel[:, 0])  # -pi 〜 pi
        _rb_dists = np.linalg.norm(_rb_rel, axis=1)
        _rb_bin_idx = np.clip(
            np.floor((_rb_angles + np.pi) / (2 * np.pi) * OUTER_BOUNDARY_N_BINS).astype(int),
            0, OUTER_BOUNDARY_N_BINS - 1,
        )

        _rb_boundary_points = []
        _rb_empty_bins = 0
        for _b in range(OUTER_BOUNDARY_N_BINS):
            _mask = _rb_bin_idx == _b
            _n_in_bin = int(np.count_nonzero(_mask))
            if _n_in_bin < MIN_BIN_POINTS:
                _rb_empty_bins += 1
                continue
            _bin_dists = _rb_dists[_mask]
            _bin_points = _rb_xz[_mask]
            _threshold = np.percentile(_bin_dists, OUTER_BOUNDARY_PERCENTILE)
            # 分位点以下でいちばん遠い点を採用する(PERCENTILE=100なら単純に最遠点)
            _candidate_idx = np.where(_bin_dists <= _threshold)[0]
            _farthest_in_candidates = _candidate_idx[np.argmax(_bin_dists[_candidate_idx])]
            _rb_boundary_points.append(_bin_points[_farthest_in_candidates])

        ROOM_BOUNDARY_POLYGON_XZ = np.array(_rb_boundary_points)  # 角度順に並んだ輪郭。以降のセルでも再利用できる
        _rb_boundary_closed = np.vstack([ROOM_BOUNDARY_POLYGON_XZ, ROOM_BOUNDARY_POLYGON_XZ[0]])


        def _polygon_area_2d(poly_xy):
            x, y = poly_xy[:, 0], poly_xy[:, 1]
            return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


        _rb_area = _polygon_area_2d(ROOM_BOUNDARY_POLYGON_XZ)
        _rb_perimeter = float(np.sum(np.linalg.norm(np.diff(_rb_boundary_closed, axis=0), axis=1)))

        print(f"[room boundary] 点群(XZ投影): {len(_rb_xz)}点")
        print(f"[room boundary] 輪郭の頂点数: {len(ROOM_BOUNDARY_POLYGON_XZ)}"
              f" / 点が無く飛ばした方向: {_rb_empty_bins}/{OUTER_BOUNDARY_N_BINS}")
        print(f"[room boundary] 面積: {_rb_area:.2f} m^2 / 周長: {_rb_perimeter:.2f} m")

        _rb_min_x, _rb_max_x = float(_rb_xz[:, 0].min()), float(_rb_xz[:, 0].max())
        _rb_min_z, _rb_max_z = float(_rb_xz[:, 1].min()), float(_rb_xz[:, 1].max())
        print(f"[room boundary] バウンディングボックス: "
              f"X=[{_rb_min_x:.2f}, {_rb_max_x:.2f}]({_rb_max_x - _rb_min_x:.2f}m) / "
              f"Z=[{_rb_min_z:.2f}, {_rb_max_z:.2f}]({_rb_max_z - _rb_min_z:.2f}m)")


        # ---- walls (元ノートブック cell 50) ----
        def _scale_for_extra_rotation(extents, target_size_whl, extra_deg):
            """Section 12.5と全く同じロジック。extra_deg(world_rotationに加えて追加で
            ヨー回転させた角度)に矛盾しないよう、軸ごとのスケールを計算する。"""
            ext_x = max(extents[0], 1e-6)
            ext_z = max(extents[2], 1e-6)
            target = np.asarray(target_size_whl, dtype=np.float64)
            target_width, target_height, target_length = target[0], target[1], target[2]
            scale = np.ones(3, dtype=np.float64)
            scale[1] = target_height / max(extents[1], 1e-6)
            if extra_deg % 180 == 0:
                scale[0] = target_length / ext_x
                scale[2] = target_width / ext_z
            else:
                scale[0] = target_width / ext_x
                scale[2] = target_length / ext_z
            return scale

        # ==========================================
        WALL_HEIGHT = 2.5   # 壁の高さ[m](床Y=0から天井までの高さ)。部屋の実際の天井高に合わせて調整する
        WALL_CAMERA_DOT_THRESHOLD = 0.5  # カメラの正面方向との内積がこの値を超える壁面だけを削除する。
                                           # 0にすると「少しでも同じ向き」なら消してしまい、必要な壁まで
                                           # 消えやすい。1に近づけるほど「ほぼ真後ろを向いている」壁だけに
                                           # 削除対象を絞れる(=消えにくくなる)
        WALL_END_MARGIN_LENGTH = 0.4  # m。上の削除処理で壁がコの字型に切れたとき、手前側の
                                        # 両端(切れ目)が短すぎることがあるので、そこから外側へ
                                        # この長さだけ延長する「マージン壁」を追加する。0にすると
                                        # 追加しない
        WALL_END_DIRECTION_SMOOTHING_DISTANCE = 0.5  # m。延長方向を決める際、切れ目から
                                        # この距離以内にある辺だけを、長さで重み付けして平均する。
                                        # 本数(固定N本)ではなく距離で区切っているのは、壁が短い
                                        # パネル1枚だけで出来ている場合に、本数基準だと隣の
                                        # (全く向きの違う)壁の辺まで平均に含めてしまうことがある
                                        # ため。近い辺だけを見るので、離れた場所の別の壁の向きに
                                        # 引っ張られにくい
        WALL_SIMPLIFY_DISTANCE = 0.05  # m。共線判定に使う距離しきい値。ある点が、その
                                       # WALL_SIMPLIFY_WINDOW個先・前の点を結んだ直線からこの距離
                                       # 未満しか離れていなければ、直線上とみなして間引く。0やNone
                                       # にすると単純化しない。
        WALL_SIMPLIFY_WINDOW = 2       # 共線判定で、直前・直後何個先の点まで見るか。1だと点群の
                                       # ノイズに弱く(隣どうしの間隔が狭すぎて、本物の角でも局所的な
                                       # ズレが小さく見えて誤って間引かれてしまう)、大きすぎると
                                       # RDP法に近づいて出窓のような小さい凹凸を見逃しやすくなる。
                                       # 2程度がバランスが良い
        WALL_SCENE_OUTPUT_PATH = os.path.join(WORK_DIR, "room_walls.glb")
        # ==========================================

        if "ROOM_BOUNDARY_POLYGON_XZ" not in locals() and "ROOM_BOUNDARY_POLYGON_XZ" not in globals():
            raise RuntimeError("ROOM_BOUNDARY_POLYGON_XZが見つかりません。先にSection 13.8を実行してください。")

        # 参考: 深度点群の高さ方向の分布(天井が写っていれば、その付近の値がWALL_HEIGHTの目安になる)
        if ("_rb_points_world" in locals() or "_rb_points_world" in globals()):
            _wall_y_values = _rb_points_world[:, 1]
            print(f"[wall] 参考: 深度点群のYの範囲 = [{_wall_y_values.min():.2f}, {_wall_y_values.max():.2f}] m"
                  f" (99パーセンタイル: {np.percentile(_wall_y_values, 99):.2f} m)")


        def _merge_collinear_points(points, distance_threshold, window=WALL_SIMPLIFY_WINDOW, max_iterations=200):
            """閉じた輪郭(points, (N,2))について、ある点が、そこから(現時点で残っている
            点どうしの並びで)window個前・window個先の点を結んだ直線からの距離が
            distance_threshold未満なら、その点を直線上とみなして間引く。

            window=1(真隣どうしだけ)だと、点群の間隔が狭い(密にサンプリングされている)
            場合、本物の角(コーナー)であってもすぐ隣の点との局所的なズレはわずかにしか
            ならず、誤って間引かれてしまうことがある。かといって遠くの点まで見てしまうと
            (RDP法のように)、出窓のような小さいが実在する凹凸まで直線の誤差の範囲内として
            消えてしまう。window=2程度の「少し先まで見る」設定にすることで、ノイズには
            ある程度強く、かつ小さな凹凸は消さないバランスを取っている。

            1点減らすごとに残りの点どうしの並び(何が「window個先」になるか)が変わるので、
            これ以上減らせなくなるまで(または max_iterations 回まで)繰り返す。
            """
            if distance_threshold is None or distance_threshold <= 0:
                return points
            keep = np.ones(len(points), dtype=bool)
            for _ in range(max_iterations):
                idxs = np.nonzero(keep)[0]
                m = len(idxs)
                if m < 2 * window + 2:
                    break
                changed = False
                for k in range(m):
                    i_prev = idxs[(k - window) % m]
                    i_cur = idxs[k]
                    i_next = idxs[(k + window) % m]
                    p_prev, p_cur, p_next = points[i_prev], points[i_cur], points[i_next]
                    line_vec = p_next - p_prev
                    line_len = np.linalg.norm(line_vec)
                    if line_len < 1e-9:
                        continue
                    line_unit = line_vec / line_len
                    proj = p_prev + np.dot(p_cur - p_prev, line_unit) * line_unit
                    dist = np.linalg.norm(p_cur - proj)
                    if dist < distance_threshold:
                        keep[i_cur] = False
                        changed = True
                if not changed:
                    break
            return points[keep]


        def _dedup_close_points(points, min_distance):
            """間引いた後、ノイズのせいで同じ角に2〜3点残ってしまうことがあるので、
            互いにmin_distance未満しか離れていない点をまとめて1点にする(閉じた輪郭の
            順番を保ったまま、近すぎる点を素通りさせる)。"""
            if min_distance is None or min_distance <= 0 or len(points) < 2:
                return points
            keep_idx = [0]
            for i in range(1, len(points)):
                if np.linalg.norm(points[i] - points[keep_idx[-1]]) >= min_distance:
                    keep_idx.append(i)
            if len(keep_idx) > 1 and np.linalg.norm(points[keep_idx[-1]] - points[keep_idx[0]]) < min_distance:
                keep_idx.pop()
            return points[keep_idx]


        WALL_COLOR_PALETTE = [
            [230, 60, 60, 150], [60, 160, 230, 150], [80, 200, 120, 150],
            [230, 180, 40, 150], [170, 90, 220, 150], [240, 120, 190, 150],
            [90, 220, 220, 150], [220, 140, 60, 150],
        ]  # 壁のパネル(辺)ごとに色を巡回させるパレット(Section 13のコライダー可視化と同じ配色)


        def _extrude_boundary_to_walls(boundary_xz, y_bottom, y_top, color_palette=None):
            """XZ平面上の閉じた輪郭(boundary_xz, (N,2)、辺の順番通りに並んでいるもの)を、
            Y方向にy_bottomからy_topまで引き延ばして、壁面(辺ごとの四角形をリング状に
            つなげたもの)のメッシュを作る。凹んだ輪郭でも、各辺は単純な四角形なので
            三角形分割に迷わない(2つの三角形に割るだけでよい)。color_paletteを渡すと、
            辺(パネル)ごとに色を巡回させて塗り分ける。"""
            n = len(boundary_xz)
            vertices = []
            faces = []
            face_colors = []
            for i in range(n):
                x0, z0 = boundary_xz[i]
                x1, z1 = boundary_xz[(i + 1) % n]
                base = len(vertices)
                vertices.append([x0, y_bottom, z0])  # base+0: 始点・床
                vertices.append([x1, y_bottom, z1])  # base+1: 終点・床
                vertices.append([x1, y_top, z1])     # base+2: 終点・天井
                vertices.append([x0, y_top, z0])     # base+3: 始点・天井
                faces.append([base + 0, base + 1, base + 2])
                faces.append([base + 0, base + 2, base + 3])
                if color_palette:
                    color = color_palette[i % len(color_palette)]
                    face_colors.append(color)
                    face_colors.append(color)
            mesh = trimesh.Trimesh(vertices=np.array(vertices), faces=np.array(faces), process=False)
            if color_palette:
                # merge_vertices()の前に頂点色ではなく面色として設定しておく
                # (面(パネル)ごとの塗り分けなので、頂点をどう共有するかとは無関係に保たれる)
                mesh.visual.face_colors = np.array(face_colors, dtype=np.uint8)
            mesh.merge_vertices()
            return mesh


        def _smoothed_end_direction(boundary, segment_keep, start_idx, step, max_distance):
            """壁の切れ目付近での「進んでいた方向」を、切れ目から近い方(距離ベース)の
            辺の向きを、辺の長さで重み付けして平均し、正規化して返す(見つからなければ
            None)。start_idxから、step(+1または-1)の向きへ辿りながら、残存している
            (segment_keep=Trueの)辺を、累積距離がmax_distanceを超えるまで集める
            (既に1本以上集まっていれば、そこで打ち切ってその辺は含めない)。

            固定本数ではなく距離で打ち切っているのは、壁が短いパネル1枚(またはごく
            少数)だけで出来ている場合に、固定本数だと隣の(全く向きの違う)壁の辺まで
            平均に含めてしまうことがあるため。近い辺だけを見ることで、離れた場所に
            ある別の壁の向きに引っ張られにくくする。
            """
            n = len(boundary)
            idx = start_idx
            dirs = []
            weights = []
            accumulated = 0.0
            while accumulated < max_distance:
                i = idx % n
                if not segment_keep[i]:
                    break
                p_a = boundary[i]
                p_b = boundary[(i + 1) % n]
                seg_vec = p_b - p_a
                seg_len = np.linalg.norm(seg_vec)
                if seg_len < 1e-9:
                    idx += step
                    continue
                if accumulated + seg_len > max_distance and dirs:
                    break  # 既に1本以上集まっていれば、これ以上遠い辺は含めない
                dirs.append(seg_vec / seg_len)
                weights.append(seg_len)
                accumulated += seg_len
                idx += step
            if not dirs:
                return None
            avg = np.average(np.array(dirs), axis=0, weights=np.array(weights))
            avg_norm = np.linalg.norm(avg)
            if avg_norm < 1e-9:
                return None
            return avg / avg_norm


        def _build_wall_quad(p0, p1, y_bottom, y_top, flipped, color=None):
            """XZ平面上の2点p0->p1を結ぶ1枚の壁パネル(四角形2枚)を作る。flipped=Trueなら
            _extrude_boundary_to_walls()側で法線を反転させたのと同じ向きに揃える。"""
            x0, z0 = p0
            x1, z1 = p1
            vertices = np.array([
                [x0, y_bottom, z0], [x1, y_bottom, z1], [x1, y_top, z1], [x0, y_top, z0],
            ])
            faces = np.array([[0, 1, 2], [0, 2, 3]])
            mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
            if flipped:
                mesh.invert()
            if color is not None:
                mesh.visual.face_colors = np.array([color, color], dtype=np.uint8)
            return mesh


        WALL_BOUNDARY_POLYGON_XZ = _merge_collinear_points(
            ROOM_BOUNDARY_POLYGON_XZ, WALL_SIMPLIFY_DISTANCE, window=WALL_SIMPLIFY_WINDOW
        )
        WALL_BOUNDARY_POLYGON_XZ = _dedup_close_points(WALL_BOUNDARY_POLYGON_XZ, WALL_SIMPLIFY_DISTANCE * 0.6)
        print(f"[wall] 輪郭を単純化しました(直線区間をまとめて間引き): "
              f"{len(ROOM_BOUNDARY_POLYGON_XZ)}点 -> {len(WALL_BOUNDARY_POLYGON_XZ)}点"
              f" (distance_threshold={WALL_SIMPLIFY_DISTANCE}m, window={WALL_SIMPLIFY_WINDOW})")

        wall_mesh = _extrude_boundary_to_walls(
            WALL_BOUNDARY_POLYGON_XZ, y_bottom=0.0, y_top=WALL_HEIGHT, color_palette=WALL_COLOR_PALETTE
        )
        # 内側(部屋の中)から見たときに面が見えるよう、法線の向きを確認して必要なら反転する。
        # 輪郭の頂点順(反時計回り/時計回り)によって面の裏表が変わるため、部屋の中心付近の
        # 点から見て法線が中心を向いている(=内側を向いている)かどうかで判定する。
        _wall_centroid_xz = ROOM_BOUNDARY_POLYGON_XZ.mean(axis=0)
        _wall_face_centers = wall_mesh.triangles_center
        _to_centroid = np.array([_wall_centroid_xz[0], _wall_face_centers[:, 1].mean(), _wall_centroid_xz[1]]) - _wall_face_centers
        _dot = np.einsum("ij,ij->i", wall_mesh.face_normals, _to_centroid)
        if np.mean(_dot) < 0:
            wall_mesh.invert()
            print("[wall] 面の向きを内側向きに反転しました。")

        # --- カメラの正面方向と法線が同じ向き(内積が正)の壁面を削除する。
        #     このワールド座標系ではカメラは原点にあり、Section 12.5と同じ変換
        #     (_R_align・_M_cv_to_gl)で、OpenCVカメラ空間の+Z(奥行き方向)をワールド座標系
        #     に変換すればカメラの正面方向が求まる。壁は内側向きの法線を持つので、
        #     「カメラの正面方向と同じ向きを向いている壁」は、カメラの後ろ側(=写真には
        #     写っておらず、輪郭を閉じるために機械的に繋いだだけの区間)にあることになる。
        #     これを取り除くことで、実際に写真に写っている手前〜奥の壁だけを残す。 ---
        _wall_R_align = globals().get("_R_align", np.eye(3))
        _wall_M_cv_to_gl = globals().get("_M_cv_to_gl", np.diag([1.0, -1.0, -1.0]))
        _camera_forward_cv = np.array([0.0, 0.0, 1.0])  # OpenCVカメラ空間での奥行き(前方)方向
        _camera_forward_world = _wall_M_cv_to_gl @ (_wall_R_align @ _camera_forward_cv)
        _camera_forward_world = _camera_forward_world / (np.linalg.norm(_camera_forward_world) + 1e-8)

        _wall_face_dot = wall_mesh.face_normals @ _camera_forward_world
        _wall_keep_mask = _wall_face_dot <= WALL_CAMERA_DOT_THRESHOLD
        _n_wall_faces_before = len(wall_mesh.faces)

        # 面は辺(パネル)ごとに2枚ずつ順番に並んでいる(_extrude_boundary_to_walls参照)ので、
        # 1つおきに見れば辺ごとのkeep判定になる(同じ辺の2枚は同一平面なので同じ判定になるはず)。
        _wall_segment_keep = _wall_keep_mask[0::2]
        _wall_was_inverted = bool(np.mean(_dot) < 0)  # 上のinvert()判定と同じ条件
        # 注意: trimeshのsubmesh()は面ごとの色(face_colors)を正しく引き継がない(頂点色に
        # 変換される際に混ざってしまう)ことがあるため、ここではfacesとface_colorsを
        # 手動で同じマスクで揃えて新しいメッシュを作る。
        _wall_kept_face_colors = wall_mesh.visual.face_colors[_wall_keep_mask].copy()
        wall_mesh = trimesh.Trimesh(
            vertices=wall_mesh.vertices, faces=wall_mesh.faces[_wall_keep_mask], process=False
        )
        wall_mesh.visual.face_colors = _wall_kept_face_colors
        print(f"[wall] カメラの正面方向とほぼ同じ向き(内積>{WALL_CAMERA_DOT_THRESHOLD})の壁面を削除しました: "
              f"{_n_wall_faces_before}面 -> {len(wall_mesh.faces)}面")

        # --- 手前側の切れ目(削除区間と残存区間の境目)の両端に、少しだけ延長した
        #     マージン壁を追加する。壁が短く途切れて見えるのを防ぐための、見た目上の
        #     延長パネル。 ---
        if WALL_END_MARGIN_LENGTH > 0:
            _n_wall_segments = len(WALL_BOUNDARY_POLYGON_XZ)
            _wall_margin_meshes = []
            for _i in range(_n_wall_segments):
                _prev_keep = _wall_segment_keep[(_i - 1) % _n_wall_segments]
                _cur_keep = _wall_segment_keep[_i]
                _seg_color = WALL_COLOR_PALETTE[_i % len(WALL_COLOR_PALETTE)]

                if _cur_keep and not _prev_keep:
                    # ここが残存区間の「始点」側の切れ目。辺i以降、直近数本の辺の向きを
                    # 平均し、その逆方向(区間の外側)へ延長する
                    _p0 = WALL_BOUNDARY_POLYGON_XZ[_i]
                    _direction = _smoothed_end_direction(
                        WALL_BOUNDARY_POLYGON_XZ, _wall_segment_keep, _i, +1,
                        WALL_END_DIRECTION_SMOOTHING_DISTANCE,
                    )
                    if _direction is None:
                        continue
                    _p_ext = _p0 - _direction * WALL_END_MARGIN_LENGTH
                    _margin_mesh = _build_wall_quad(
                        _p_ext, _p0, 0.0, WALL_HEIGHT, flipped=_wall_was_inverted, color=_seg_color
                    )
                    _wall_margin_meshes.append(_margin_mesh)

                if not _cur_keep and _prev_keep:
                    # ここが残存区間の「終点」側の切れ目。辺i-1以前、直近数本の辺の向きを
                    # 平均し、そのまま延長する
                    _prev_seg_color = WALL_COLOR_PALETTE[(_i - 1) % _n_wall_segments % len(WALL_COLOR_PALETTE)]
                    _p_end = WALL_BOUNDARY_POLYGON_XZ[_i]
                    _direction = _smoothed_end_direction(
                        WALL_BOUNDARY_POLYGON_XZ, _wall_segment_keep, (_i - 1) % _n_wall_segments, -1,
                        WALL_END_DIRECTION_SMOOTHING_DISTANCE,
                    )
                    if _direction is None:
                        continue
                    _p_ext = _p_end + _direction * WALL_END_MARGIN_LENGTH
                    _margin_mesh = _build_wall_quad(
                        _p_end, _p_ext, 0.0, WALL_HEIGHT, flipped=_wall_was_inverted, color=_prev_seg_color
                    )
                    _wall_margin_meshes.append(_margin_mesh)

            if _wall_margin_meshes:
                wall_mesh = trimesh.util.concatenate([wall_mesh] + _wall_margin_meshes)
                print(f"[wall] 手前側の切れ目に、長さ{WALL_END_MARGIN_LENGTH}mのマージン壁を"
                      f"{len(_wall_margin_meshes)}枚追加しました。")

        print(f"[wall] 壁メッシュ: 頂点数={len(wall_mesh.vertices)}, 面数={len(wall_mesh.faces)}, "
              f"高さ={WALL_HEIGHT}m")

        # 面ごとの色は_extrude_boundary_to_walls()で既に設定済み(WALL_COLOR_PALETTEで
        # パネルごとに塗り分け)。ここで上書きはしない。

        # --- 壁メッシュだけをシーンに入れて出力する ---
        #     (以前はここに深度点群と配置済みの家具モデルも一緒に追加していたが、
        #      room_walls.glb には壁以外のデータを含めないため削除した。
        #      Section 13.5の壁との衝突判定はグローバル変数 wall_mesh を直接使うので影響しない。)
        wall_scene = trimesh.Scene()
        wall_scene.add_geometry(wall_mesh, node_name="room_walls")

        wall_scene.export(WALL_SCENE_OUTPUT_PATH)
        print("saved:", WALL_SCENE_OUTPUT_PATH)


        progress({"percent": _PCT["coacd"], "label": "物理コライダー生成中"})

        # ---- coacd (元ノートブック cell 52) ----

        import coacd
        import sys
        import multiprocessing
        import importlib
        from concurrent.futures import ProcessPoolExecutor, as_completed

        # ==========================================
        # CoACDのパラメータを、家具のサイズ(OBBの体積、幅×高さ×奥行き[m^3])に応じて
        # 3段階に変える。観葉植物の葉のように「多少おおざっぱでも構わない」小さい物体には
        # 粗い(速い)設定を、大きく物理的な存在感のある家具にはこれまで通り丁寧めの設定を使う。
        # COACD_FAST_MODE: True にすると、mcts探索(②)と解像度(③)を下げた「速度優先」設定を使う。
        # コライダーの形はやや粗くなる(特に凹凸の細かい家具で、凹み部分の再現が甘くなることがある)。
        # False にすれば、いつでも元の設定に戻せる。
        # (この関数の冒頭で options["coacd_fast_mode"] から決めた値を、そのまま使う。
        #  ここで固定値に上書きしない。)

        if COACD_FAST_MODE:
            COACD_SIZE_TIERS = [
                # (この体積[m^3]未満ならこの段階を適用, パラメータ辞書)
                (0.02, {  # 小物(観葉植物・置物・小さい装飾品など): とにかく速く、粗くて良い
                    "threshold": 0.5, "max_faces": 500,
                    "mcts_nodes": 4, "mcts_iterations": 10, "mcts_max_depth": 1,
                    "preprocess_resolution": 16, "resolution": 350,
                }),
                (0.3, {  # 中型家具(椅子・小さめのテーブルなど)
                    "threshold": 0.3, "max_faces": 2000,
                    "mcts_nodes": 6, "mcts_iterations": 30, "mcts_max_depth": 1,
                    "preprocess_resolution": 22, "resolution": 700,
                }),
                (float("inf"), {  # 大型家具(ソファ・ベッド・棚など): 個数・存在感が大きいので少し丁寧に
                    "threshold": 0.3, "max_faces": 3000,
                    "mcts_nodes": 8, "mcts_iterations": 50, "mcts_max_depth": 2,
                    "preprocess_resolution": 28, "resolution": 1000,
                }),
            ]
        else:
            COACD_SIZE_TIERS = [
                # (この体積[m^3]未満ならこの段階を適用, パラメータ辞書)
                (0.02, {  # 小物(観葉植物・置物・小さい装飾品など): とにかく速く、粗くて良い
                    "threshold": 0.5, "max_faces": 500,
                    "mcts_nodes": 8, "mcts_iterations": 20, "mcts_max_depth": 1,
                    "preprocess_resolution": 20, "resolution": 500,
                }),
                (0.3, {  # 中型家具(椅子・小さめのテーブルなど)
                    "threshold": 0.3, "max_faces": 2000,
                    "mcts_nodes": 12, "mcts_iterations": 60, "mcts_max_depth": 2,
                    "preprocess_resolution": 30, "resolution": 1000,
                }),
                (float("inf"), {  # 大型家具(ソファ・ベッド・棚など): 個数・存在感が大きいので少し丁寧に
                    "threshold": 0.3, "max_faces": 3000,
                    "mcts_nodes": 16, "mcts_iterations": 100, "mcts_max_depth": 3,
                    "preprocess_resolution": 40, "resolution": 1500,
                }),
            ]
        print("COACD_FAST_MODE:", COACD_FAST_MODE)
        COACD_DECIMATE_AGGRESSIVENESS = 7  # pyfqmrの削減の積極度(0〜10。大きいほど速いが荒い簡略化。全段階共通)
        # CoACDはCPUバウンドなので、家具ごとにプロセス並列で実行する。
        # os.cpu_count()ではなく、実際にこのコンテナへ割り当てられたCPU数(sched_getaffinity)を使う。
        # 上限は特に設けない(vCPUが多いPodなら、そのぶんまとめて並列化する)。
        COACD_WORKERS = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else (os.cpu_count() or 1)
        COLLISION_DOWNLOAD_DIR = os.path.join(WORK_DIR, "_tmp_collision_objects")
        COLLISION_SCENE_OUTPUT_PATH = os.path.join(WORK_DIR, "collision_coacd.glb")
        # ==========================================


        def _coacd_params_for_size(size_whl):
            """OBBの体積(幅×高さ×奥行き、m^3)から、CoACD_SIZE_TIERSに沿ってパラメータを選ぶ。"""
            volume = float(size_whl[0]) * float(size_whl[1]) * float(size_whl[2])
            for max_volume, params in COACD_SIZE_TIERS:
                if volume < max_volume:
                    return volume, params
            return volume, COACD_SIZE_TIERS[-1][1]

        # --- 前回このセルを実行した際にダウンロードしたファイルを削除してから始める ---
        if os.path.isdir(COLLISION_DOWNLOAD_DIR):
            shutil.rmtree(COLLISION_DOWNLOAD_DIR, ignore_errors=True)
        os.makedirs(COLLISION_DOWNLOAD_DIR, exist_ok=True)

        # --- Section 12.5で配置した家具と同じモデルをダウンロードし直す(Section 12.5の末尾で
        #     ダウンロード済みファイルは削除済みのため) ---
        _collision_uid_to_path = _download_hssd_objects_to(
            sorted({e["uid"] for e in _placement_entries}), COLLISION_DOWNLOAD_DIR
        )
        print(f"ダウンロード完了: {len(_collision_uid_to_path)} / {len(_placement_entries)}")

        # --- ワーカー関数を独立した.pyファイルとして書き出し、importする。
        #     ProcessPoolExecutorの既定(fork)は、このノートブックが同じカーネルの中で
        #     PyTorch/CUDAを初期化済みのため、子プロセスが不正なCUDA状態やJupyterの出力用
        #     ロックを引き継いでフリーズ(ハング)することがある。これを避けるため、
        #     子プロセスを何も引き継がずまっさらな状態で起動する`spawn`方式を使う。
        #     `spawn`は関数を実体(モジュール内の名前)としてpickleするため、ノートブックの
        #     セル内で定義した関数(実体は__main__にしかない)は使えず、独立したモジュールに
        #     切り出す必要がある。 ---
        _coacd_worker_path = os.path.join(WORK_DIR, "_coacd_worker.py")
        _coacd_worker_source = r'''
"""CoACD分解を1家具ぶん行うワーカー関数(spawn方式のProcessPoolExecutorから呼ばれる)。"""
import coacd
import numpy as np
import pyfqmr
import trimesh


def _decimate_with_pyfqmr(vertices, faces, target_count, aggressiveness):
    simplifier = pyfqmr.Simplify()
    simplifier.setMesh(vertices, faces)
    simplifier.simplify_mesh(
        target_count=int(target_count),
        aggressiveness=aggressiveness,
        preserve_border=True,
        verbose=False,
    )
    new_vertices, new_faces, _normals = simplifier.getMesh()
    return new_vertices, new_faces


def _scale_for_extra_rotation(extents, target_size_whl, extra_deg):
    """Section 12.5と全く同じロジック。extra_deg(world_rotationに加えて追加で
    ヨー回転させた角度)に矛盾しないよう、軸ごとのスケールを計算する。"""
    ext_x = max(extents[0], 1e-6)
    ext_z = max(extents[2], 1e-6)
    target = np.asarray(target_size_whl, dtype=np.float64)
    target_width, target_height, target_length = target[0], target[1], target[2]
    scale = np.ones(3, dtype=np.float64)
    scale[1] = target_height / max(extents[1], 1e-6)
    if extra_deg % 180 == 0:
        scale[0] = target_length / ext_x
        scale[2] = target_width / ext_z
    else:
        scale[0] = target_width / ext_x
        scale[2] = target_length / ext_z
    return scale


def _build_placement_transform(local_center, scale_xyz, rotation_matrix_3x3, world_position):
    T_center = np.eye(4)
    T_center[:3, 3] = -np.asarray(local_center, dtype=np.float64)

    S = np.eye(4)
    S[0, 0], S[1, 1], S[2, 2] = scale_xyz

    R = np.eye(4)
    R[:3, :3] = rotation_matrix_3x3

    T_place = np.eye(4)
    T_place[:3, 3] = np.asarray(world_position, dtype=np.float64)

    return T_place @ R @ S @ T_center


def run_coacd_for_one(job):
    (entry_index, uid, category, local_path, size_whl, world_rotation, world_position,
     mesh_extra_yaw_deg,
     coacd_threshold, coacd_max_faces, coacd_decimate_aggressiveness,
     coacd_preprocess_resolution, coacd_resolution,
     coacd_mcts_nodes, coacd_mcts_iterations, coacd_mcts_max_depth) = job

    mesh = trimesh.load(local_path, force="mesh")

    # world_rotationはSection 12.5で書き込まれた、正面補正・形状補正込みの最終的な
    # 回転。mesh_extra_yaw_degはその際に選んだ追加ヨー回転角度(Section 12.5参照)。
    # 同じ角度を使ってスケールを計算することで、world_rotationと矛盾しないように
    # している。
    local_center = mesh.bounds.mean(axis=0)
    extents = mesh.bounds[1] - mesh.bounds[0]
    scale_xyz = _scale_for_extra_rotation(extents, size_whl, mesh_extra_yaw_deg)
    transform = _build_placement_transform(local_center, scale_xyz, world_rotation, world_position)

    coacd_vertices, coacd_faces = mesh.vertices, mesh.faces
    if len(coacd_faces) > coacd_max_faces:
        coacd_vertices, coacd_faces = _decimate_with_pyfqmr(
            coacd_vertices, coacd_faces, coacd_max_faces, coacd_decimate_aggressiveness
        )

    # --- CoACD自体が例外を出したり、空の結果を返したりすることがある(非多様体が
    #     ひどい・タイムアウトなど)。その場合にこの家具のコライダーが1つも無いままだと、
    #     以後のセル(めり込み解消・浮遊解消)からは「この場所には何も無い透明な家具」
    #     として扱われてしまい、他の家具がこれを支えとして見つけられなくなる。
    #     そうならないよう、失敗時は元メッシュの凸包(convex hull)1個を
    #     フォールバックのコライダーとして使う(粗いが、無いよりはるかに良い)。 ---
    try:
        coacd_mesh = coacd.Mesh(coacd_vertices, coacd_faces)
        parts = coacd.run_coacd(
            coacd_mesh,
            threshold=coacd_threshold,
            preprocess_resolution=coacd_preprocess_resolution,
            resolution=coacd_resolution,
            mcts_nodes=coacd_mcts_nodes,
            mcts_iterations=coacd_mcts_iterations,
            mcts_max_depth=coacd_mcts_max_depth,
        )
        if not parts:
            raise ValueError("coacd.run_coacd returned no parts")
    except Exception:
        fallback_mesh = trimesh.Trimesh(vertices=coacd_vertices, faces=coacd_faces, process=False)
        hull = fallback_mesh.convex_hull
        parts = [(hull.vertices, hull.faces)]

    part_arrays = []
    for part_vertices, part_faces in parts:
        part_mesh = trimesh.Trimesh(vertices=part_vertices, faces=part_faces, process=False)
        part_mesh.apply_transform(transform)
        part_arrays.append((part_mesh.vertices, part_mesh.faces))

    return entry_index, uid, category, part_arrays
        '''
        with open(_coacd_worker_path, "w", encoding="utf-8") as f:
            f.write(_coacd_worker_source)

        if WORK_DIR not in sys.path:
            sys.path.insert(0, WORK_DIR)
        import _coacd_worker
        importlib.reload(_coacd_worker)  # 前回このセルを実行した際の内容がキャッシュされていないようにする


        _coacd_jobs = []
        for _entry_index, e in enumerate(_placement_entries):
            local_path = _collision_uid_to_path.get(e["uid"])
            if not local_path:
                print(f"  [WARN] {e['uid']} のダウンロードに失敗しているためスキップ")
                continue
            volume, params = _coacd_params_for_size(e["size_whl"])
            print(f"  '{e['category']}' ({e['uid']}): 体積={volume:.3f}m^3 -> "
                  f"threshold={params['threshold']}, max_faces={params['max_faces']}, "
                  f"mcts=(nodes={params['mcts_nodes']}, iter={params['mcts_iterations']}, "
                  f"depth={params['mcts_max_depth']})")
            _coacd_jobs.append((
                _entry_index, e["uid"], e["category"], local_path, e["size_whl"],
                e["world_rotation"], e["world_position"],
                e.get("mesh_extra_yaw_deg", 0),
                params["threshold"], params["max_faces"], COACD_DECIMATE_AGGRESSIVENESS,
                params["preprocess_resolution"], params["resolution"],
                params["mcts_nodes"], params["mcts_iterations"], params["mcts_max_depth"],
            ))

        collision_results = []  # 各要素: {"entry_index", "uid", "category", "num_parts", "collider_meshes"(ワールド座標のtrimeshのリスト)}

        # ワーカー数は、CPUの上限と「家具の個数」の小さいほうにする(物体が数個しかないのに、
        # 使わないプロセスまで大量に起動して、spawn分の起動コストだけ無駄にするのを防ぐ)。
        _coacd_pool_workers = max(1, min(COACD_WORKERS, len(_coacd_jobs)))
        print(f"CoACD並列数: {_coacd_pool_workers}(CPU上限={COACD_WORKERS} / 家具数={len(_coacd_jobs)})")

        _spawn_ctx = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(max_workers=_coacd_pool_workers, mp_context=_spawn_ctx) as pool:
            _futures = {pool.submit(_coacd_worker.run_coacd_for_one, job): job for job in _coacd_jobs}
            for future in as_completed(_futures):
                job = _futures[future]
                job_entry_index, job_uid, job_category = job[0], job[1], job[2]
                try:
                    entry_index, uid, category, part_arrays = future.result()
                    collider_meshes = [
                        trimesh.Trimesh(vertices=v, faces=f, process=False) for v, f in part_arrays
                    ]
                    collision_results.append({
                        "entry_index": entry_index,
                        "uid": uid,
                        "category": category,
                        "num_parts": len(collider_meshes),
                        "collider_meshes": collider_meshes,
                    })
                    print(f"  '{category}' ({uid}): {len(collider_meshes)}個の凸パーツに分解")
                except Exception as ex:
                    print(f"  [WARN] CoACD分解失敗: {job_uid} ({job_category}): {ex!r}")

        _total_parts = sum(r["num_parts"] for r in collision_results)
        print(f"\n分解完了: {len(collision_results)} / {len(_placement_entries)}件"
              f" (合計 {_total_parts}個の凸パーツ)")

        # --- ダウンロード済みファイルは、コライダー生成が終わったら削除する ---
        shutil.rmtree(COLLISION_DOWNLOAD_DIR, ignore_errors=True)
        print(f"ダウンロード済みモデルを削除しました: {COLLISION_DOWNLOAD_DIR}")


        # --- 可視化: 家具ごとに分解されたパーツを、パーツごとに色を変えて重ねて描画する ---
        _PART_COLOR_PALETTE = [
            [230, 60, 60, 150], [60, 160, 230, 150], [80, 200, 120, 150],
            [230, 180, 40, 150], [170, 90, 220, 150], [240, 120, 190, 150],
            [90, 220, 220, 150], [220, 140, 60, 150],
        ]

        collision_scene = trimesh.Scene()
        for r in collision_results:
            for i, part_mesh in enumerate(r["collider_meshes"]):
                vis_mesh = part_mesh.copy()
                vis_mesh.visual.face_colors = _PART_COLOR_PALETTE[i % len(_PART_COLOR_PALETTE)]
                collision_scene.add_geometry(vis_mesh, node_name=f"collider_{r['category']}_{r['uid']}_{i}")

        if len(collision_scene.geometry) > 0:
            collision_scene.export(COLLISION_SCENE_OUTPUT_PATH)
            print("saved:", COLLISION_SCENE_OUTPUT_PATH)
        else:
            print("分解できた家具がありませんでした。")


        # ---- coacd_simplify (元ノートブック cell 56) ----

        # ==========================================
        OBB_COLLIDER_VOLUME_RATIO_THRESHOLD = 0.85  # コライダー(CoACDの凸パーツ)の体積が、
                                                      # OBBの体積のこの割合以上を占めていれば、
                                                      # OBBそのものの単純な直方体コライダーに
                                                      # 置き換える。1に近づけるほど「ほぼ完全に
                                                      # 箱型」でないと置き換えなくなる
        UNION_VOLUME_MONTE_CARLO_SAMPLES = 8000     # 凸パーツどうしの重なりを考慮した和集合の
                                                      # 体積を、モンテカルロ法(点のサンプリング)で
                                                      # 近似する際の点の数。多いほど精度が上がるが遅くなる
        # ==========================================


        def _obb_volume(size_whl):
            return float(size_whl[0]) * float(size_whl[1]) * float(size_whl[2])


        def _mesh_volume_safe(mesh):
            try:
                return float(abs(mesh.volume))
            except Exception:
                return 0.0


        def _collider_union_volume(meshes, n_samples=UNION_VOLUME_MONTE_CARLO_SAMPLES, seed=0):
            """複数の凸パーツの体積を単純に合計すると、パーツどうしが重なっている部分を
            二重に数えてしまい、実際より大きい値になる(本来は箱型でない家具まで、重なりの
            せいで誤って「箱型に近い」と判定されてしまう原因になる)。ここでは和集合
            (union)の体積を、モンテカルロ法(点のランダムサンプリング)で近似する:
            全パーツを囲むバウンディングボックス内に点をランダムに撒き、いずれか1つの
            パーツの内部に入っている点の割合から体積を推定する。パーツが1つだけなら
            重なりが起きないので、直接mesh.volumeを使う(サンプリングより速く正確)。"""
            if not meshes:
                return 0.0
            if len(meshes) == 1:
                return _mesh_volume_safe(meshes[0])

            all_bounds = np.array([m.bounds for m in meshes])
            box_min = all_bounds[:, 0, :].min(axis=0)
            box_max = all_bounds[:, 1, :].max(axis=0)
            box_volume = float(np.prod(box_max - box_min))
            if box_volume < 1e-9:
                return 0.0

            rng = np.random.default_rng(seed)
            points = rng.uniform(box_min, box_max, size=(n_samples, 3))
            inside = np.zeros(n_samples, dtype=bool)
            for m in meshes:
                try:
                    inside |= m.contains(points)
                except Exception:
                    continue
            fraction = float(inside.sum()) / n_samples
            return fraction * box_volume


        def _make_obb_box_collider(size_whl, world_rotation, world_position):
            """OBBの寸法・向き・位置に完全に一致する直方体コライダーを1つ作る。
            ローカルX軸=長さ(l)、Y軸=高さ(h)、Z軸=幅(w)という規約(Section 7.6の
            _obb_corners_camと同じ)に沿っている。"""
            width, height, length = size_whl
            box = trimesh.creation.box(extents=[length, height, width])
            transform = np.eye(4)
            transform[:3, :3] = world_rotation
            transform[:3, 3] = world_position
            box.apply_transform(transform)
            return box


        _n_replaced_with_box = 0
        _n_skipped_cheaply = 0
        for r in collision_results:
            e = _placement_entries[r["entry_index"]]
            obb_volume = _obb_volume(e["size_whl"])
            if obb_volume < 1e-9:
                continue

            # --- 高速化: 和集合(union)の体積は、単純な合計(重なりを二重に数えたもの)
            #     より必ず小さいか同じになる。なので、まず安価な単純合計を計算し、その
            #     比率が既に閾値未満なら、正確な和集合を計算してもどのみち閾値未満の
            #     まま(=箱型ではないと確定)なので、重いモンテカルロ計算を省略できる。
            #     モンテカルロ計算が必要なのは、単純合計の比率が閾値以上になった
            #     (=重なりを除いても本当に閾値以上か確認が必要な)場合だけ。 ---
            _naive_sum = sum(_mesh_volume_safe(m) for m in r["collider_meshes"])
            _naive_ratio = _naive_sum / obb_volume
            if _naive_ratio < OBB_COLLIDER_VOLUME_RATIO_THRESHOLD:
                _n_skipped_cheaply += 1
                continue

            collider_volume = _collider_union_volume(r["collider_meshes"])
            ratio = collider_volume / obb_volume

            if ratio >= OBB_COLLIDER_VOLUME_RATIO_THRESHOLD:
                box_collider = _make_obb_box_collider(e["size_whl"], e["world_rotation"], e["world_position"])
                r["collider_meshes"] = [box_collider]
                _n_replaced_with_box += 1
                print(f"  '{e['category']}'({e['uid']}): コライダー体積/OBB体積={ratio:.2f}"
                      f" -> OBBそのものの直方体コライダーに置き換えました")

        print(f"\n{_n_replaced_with_box} / {len(collision_results)}件を単純な直方体コライダーに置き換えました"
              f"(閾値: 体積比{OBB_COLLIDER_VOLUME_RATIO_THRESHOLD}以上)"
              f" / {_n_skipped_cheaply}件は単純合計の時点で閾値未満と分かり、モンテカルロ計算を省略")


        # ---- placement_fix (元ノートブック cell 58) ----

        # --- trimeshは既にこのノートブックの前の方のセルでimport済みだが、その時点では
        #     python-fclがまだ入っていなかったため、trimesh.collisionモジュール内部で
        #     「fcl = None(利用不可)」と記憶されてしまっている。上のpip installで後から
        #     python-fclを入れても、import済みのモジュールの中身は自動更新されないため、
        #     明示的にreloadしてtrimesh.collisionにfclを再認識させる。 ---
        import importlib
        importlib.reload(trimesh.collision)
        from scipy.spatial import ConvexHull
        from scipy.cluster.vq import kmeans2
        import matplotlib.colors as mcolors

        # ==========================================
        # --- Step0: 大小分け ---
        SMALL_OBJECT_VOLUME_THRESHOLD = 0.1    # m^3。OBBの体積(幅x高さx奥行き)がこれ未満なら「小さいオブジェクト」
        FORCE_SMALL_CATEGORIES = {"tv", "computer monitor"}
           # 体積が閾値以上でも、カテゴリ名がこの集合に完全一致すれば強制的に
           # 「小さいオブジェクト」(=どこかの親に乗る子オブジェクト)として扱う。
           # テレビのように、大きくてもほぼ必ず台の上に乗るものを想定。
           # 注意: 部分一致(in)だと"tv stand"のような紛らわしいカテゴリ名まで誤って
           # マッチしてしまう("tv stand"は逆に他の物を乗せる側の「大きいオブジェクト」で
           # あるべき)ため、完全一致にしている。DETECTION_PROMPTSのカテゴリ名(小文字)と
           # 一致するように指定すること。

        # --- Step1/2: 小さいオブジェクトの設置先探索・親子付け ---
        LANDING_UP_MIN_DOT = 0.9           # 面の法線とワールドの上向き[0,1,0]との内積がこれ以上なら「上向き」とみなす
        LANDING_MIN_OVERLAP_RATIO = 0.8    # OBB底面の面積のうち、この割合以上が面と重なっていれば「収まる」とみなす
        CHILD_SEARCH_RADIUS = 3.0          # 設置先を探す水平方向の最大距離[m]
        CHILD_MAX_HEIGHT_DIFF = 1.5        # 設置先の高さと現在のOBB底面の高さの差がこれを超える候補は除外[m]
        CHILD_HEIGHT_WEIGHT = 10.0         # スコアリングで高さの差を重視する度合い(距離より大きく)
        CHILD_DISTANCE_WEIGHT = 1.0        # スコアリングで水平距離を考慮する度合い
        CHILD_SEARCH_STEPS = 20            # 「収まる位置」を直線的に探す際の刻み数

        # --- Step3: 親(非子オブジェクト)の接地 ---
        FLOOR_PLANE_SIZE = 100.0           # 床の代わりに使う、十分に大きい仮想平面のサイズ[m]
        GROUND_CONTACT_TOLERANCE = 0.05    # 底面が既にこの範囲[m]以内でY=0に接していれば接地処理をスキップ
        SUPPORT_MUST_BE_BIGGER_FACTOR = 1.2  # 床以外の「大きいオブジェクト」を接地レイの対象に含める条件。
                                            # 自分自身のOBB体積のこの倍以上ある大きいオブジェクトだけを対象にする
                                            # (OBB推定の誤差でたまたま体積が大きく出て「大きいオブジェクト」に
                                            # 分類されてしまった、本来は小物であるはずの物体が、床に直接
                                            # 置かれてしまうのを防ぐため。1.0にすると「自分より少しでも大きい」
                                            # だけで対象になる。Noneにすると床のみに戻る)

        # --- Step4: 親どうしのめり込み解消(Section 13.5の元のロジックと同じ) ---
        OVERLAP_MAX_ITERATIONS = 60
        OVERLAP_RELAXATION = 0.5
        OVERLAP_CONVERGENCE_TOLERANCE = 0.002

        # --- 色合わせ(Section 12.5と同じロジック。最終エクスポート時に使う) ---
        RECOLOR_TO_MATCH_PHOTO = True
        RECOLOR_HUE_STRENGTH = 0.85
        RECOLOR_SATURATION_STRENGTH = 0.5
        RECOLOR_N_COLOR_CLUSTERS = 3
        RECOLOR_MIN_TARGET_SATURATION = 0.12

        FIXED_SCENE_OUTPUT_PATH = os.path.join(WORK_DIR, "scene_reconstruction_fixed.glb")
        FIXED_COLLIDER_SCENE_OUTPUT_PATH = os.path.join(WORK_DIR, "collision_coacd_fixed.glb")
        # ==========================================


        # ============================================================
        # ヘルパー関数群(設置可能面の抽出・重なり判定は Section 13.6/13.7 と同じ考え方)
        # ============================================================

        def _scale_for_extra_rotation(extents, target_size_whl, extra_deg):
            """Section 12.5と全く同じロジック。extra_deg(world_rotationに加えて追加で
            ヨー回転させた角度)に矛盾しないよう、軸ごとのスケールを計算する。"""
            ext_x = max(extents[0], 1e-6)
            ext_z = max(extents[2], 1e-6)
            target = np.asarray(target_size_whl, dtype=np.float64)
            target_width, target_height, target_length = target[0], target[1], target[2]
            scale = np.ones(3, dtype=np.float64)
            scale[1] = target_height / max(extents[1], 1e-6)
            if extra_deg % 180 == 0:
                scale[0] = target_length / ext_x
                scale[2] = target_width / ext_z
            else:
                scale[0] = target_width / ext_x
                scale[2] = target_length / ext_z
            return scale


        def _obb_volume(size_whl):
            return float(size_whl[0]) * float(size_whl[1]) * float(size_whl[2])


        def _obb_footprint_corners_xz(size_whl, world_rotation):
            """OBB底面の4隅の、ワールド空間(XZ)でのオフセットベクトルを、四角形として
            正しく一周する順番で返す。ローカルX軸=長さ(l)、ローカルZ軸=幅(w)という規約
            (Section 7.6の_obb_corners_camで確認済み)に沿っている。"""
            width, _height, length = size_whl
            half_x = float(length) / 2.0
            half_z = float(width) / 2.0
            x_axis = world_rotation[:, 0]
            z_axis = world_rotation[:, 2]
            corners_3d = [
                half_x * x_axis + half_z * z_axis,
                half_x * x_axis - half_z * z_axis,
                -half_x * x_axis - half_z * z_axis,
                -half_x * x_axis + half_z * z_axis,
            ]
            return np.array([[c[0], c[2]] for c in corners_3d])


        def _extract_upward_face_groups(mesh, min_dot=LANDING_UP_MIN_DOT):
            """メッシュの中から法線が上向き(min_dot以上)な面を取り出し、面同士のつながりで
            グループ化した(連結成分ごとの)サブメッシュのリストを返す。1つの連結成分 =
            「途切れずにつながった1枚の設置可能面」。"""
            if len(mesh.faces) == 0:
                return []
            normals = mesh.face_normals
            mask = normals[:, 1] >= min_dot
            face_indices = np.nonzero(mask)[0]
            if len(face_indices) == 0:
                return []
            upward_sub = mesh.submesh([face_indices], append=True)
            if len(upward_sub.faces) == 0:
                return []
            components = trimesh.graph.connected_components(upward_sub.face_adjacency, min_len=1)
            if len(components) == 0:
                return [upward_sub]
            groups = []
            for comp_face_idx in components:
                group_mesh = upward_sub.submesh([comp_face_idx], append=True)
                if len(group_mesh.faces) > 0:
                    groups.append(group_mesh)
            return groups


        def _surface_descriptor(group_mesh, entry_index):
            """設置可能面(連結成分)1枚から、XZ平面上の凸包と、高さを推定するための
            平面近似(最小二乗)を作る。"""
            verts = group_mesh.vertices
            xz = verts[:, [0, 2]]
            if len(xz) < 3:
                return None
            try:
                hull = ConvexHull(xz)
            except Exception:
                return None
            polygon = xz[hull.vertices]
            A = np.column_stack([verts[:, 0], verts[:, 2], np.ones(len(verts))])
            coeffs, *_ = np.linalg.lstsq(A, verts[:, 1], rcond=None)
            return {
                "entry_index": entry_index,
                "polygon": polygon,
                "centroid_xz": polygon.mean(axis=0),
                "plane": coeffs,
            }


        def _polygon_area(poly):
            if len(poly) < 3:
                return 0.0
            x, y = poly[:, 0], poly[:, 1]
            return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


        def _polygon_signed_area(poly):
            x, y = poly[:, 0], poly[:, 1]
            return 0.5 * (np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


        def _clip_convex_polygon(subject, clip):
            """Sutherland-Hodgmanアルゴリズムで、凸多角形subjectを凸多角形clipで切り取り、
            その交差部分の頂点列を返す。"""
            if len(subject) < 3 or len(clip) < 3:
                return np.zeros((0, 2))
            if _polygon_signed_area(subject) < 0:
                subject = subject[::-1]
            if _polygon_signed_area(clip) < 0:
                clip = clip[::-1]

            def _is_inside(p, a, b):
                return (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0]) >= -1e-9

            def _line_intersect(p1, p2, a, b):
                A1, B1 = b[1] - a[1], a[0] - b[0]
                C1 = A1 * a[0] + B1 * a[1]
                A2, B2 = p2[1] - p1[1], p1[0] - p2[0]
                C2 = A2 * p1[0] + B2 * p1[1]
                det = A1 * B2 - A2 * B1
                if abs(det) < 1e-12:
                    return p2
                x = (B2 * C1 - B1 * C2) / det
                y = (A1 * C2 - A2 * C1) / det
                return np.array([x, y])

            output = list(subject)
            for i in range(len(clip)):
                a, b = clip[i], clip[(i + 1) % len(clip)]
                if not output:
                    break
                input_list, output = output, []
                for j in range(len(input_list)):
                    cur, prev = input_list[j], input_list[j - 1]
                    cur_in, prev_in = _is_inside(cur, a, b), _is_inside(prev, a, b)
                    if cur_in:
                        if not prev_in:
                            output.append(_line_intersect(prev, cur, a, b))
                        output.append(cur)
                    elif prev_in:
                        output.append(_line_intersect(prev, cur, a, b))
            return np.array(output) if output else np.zeros((0, 2))


        def _footprint_overlap_ratio(surface_polygon, footprint_corners_xz):
            footprint_area = _polygon_area(footprint_corners_xz)
            if footprint_area < 1e-9:
                return 0.0
            inter = _clip_convex_polygon(footprint_corners_xz, surface_polygon)
            return _polygon_area(inter) / footprint_area


        def _plane_height(surface, x, z):
            a, b, c = surface["plane"]
            return float(a * x + b * z + c)


        def _nearest_point_in_convex_polygon(point, polygon):
            poly = polygon
            if _polygon_signed_area(poly) < 0:
                poly = poly[::-1]
            n = len(poly)
            inside = True
            for i in range(n):
                a, b = poly[i], poly[(i + 1) % n]
                cross = (b[0] - a[0]) * (point[1] - a[1]) - (b[1] - a[1]) * (point[0] - a[0])
                if cross < -1e-9:
                    inside = False
                    break
            if inside:
                return point.copy()
            best_pt, best_dist = None, None
            for i in range(n):
                a, b = poly[i], poly[(i + 1) % n]
                ab = b - a
                denom = np.dot(ab, ab)
                t = np.dot(point - a, ab) / denom if denom > 1e-12 else 0.0
                t = np.clip(t, 0.0, 1.0)
                proj = a + t * ab
                dist = np.linalg.norm(point - proj)
                if best_dist is None or dist < best_dist:
                    best_dist, best_pt = dist, proj
            return best_pt


        def _search_nearby_fit(current_xz, corner_offsets_xz, surface, min_ratio, max_distance,
                                steps=CHILD_SEARCH_STEPS):
            """current_xzから、その面の中で「いちばん近い点」に向かって直線的に進み、
            OBB底面が最初にmin_ratio以上収まった時点の位置を返す(見つからなければNone)。"""
            target_xz = _nearest_point_in_convex_polygon(current_xz, surface["polygon"])
            direction = target_xz - current_xz
            dist_to_target = float(np.linalg.norm(direction))
            if dist_to_target < 1e-9:
                direction = surface["centroid_xz"] - current_xz
                dist_to_target = float(np.linalg.norm(direction))
                if dist_to_target < 1e-9:
                    return None
            unit = direction / dist_to_target
            for i in range(1, steps + 1):
                dist = max_distance * i / steps
                candidate_xz = current_xz + unit * dist
                corners_xz = candidate_xz + corner_offsets_xz
                ratio = _footprint_overlap_ratio(surface["polygon"], corners_xz)
                if ratio >= min_ratio:
                    return candidate_xz
            return None


        # ============================================================
        # Step0: 大きいオブジェクト/小さいオブジェクトに分ける
        # ============================================================
        _collider_lookup = {r["entry_index"]: r["collider_meshes"] for r in collision_results}
        _uncovered_entries = [i for i in range(len(_placement_entries)) if i not in _collider_lookup]
        if _uncovered_entries:
            print(f"  [WARN] {len(_uncovered_entries)}件はSection 13でコライダーが作れなかったため、"
                  "このセクションの対象外です(位置はそのまま)。")

        def _is_force_small_category(category):
            return category.lower().strip() in FORCE_SMALL_CATEGORIES


        _large_indices = []
        _small_indices = []
        _n_force_small = 0
        for idx, e in enumerate(_placement_entries):
            if idx not in _collider_lookup:
                continue
            _vol = _obb_volume(e["size_whl"])
            if _is_force_small_category(e["category"]):
                _small_indices.append(idx)
                _n_force_small += 1
                print(f"  [Step0] '{e['category']}'({e['uid']}): 体積={_vol:.3f}m^3 -> 小さいオブジェクト"
                      f"(カテゴリ名による強制指定)")
            elif _vol < SMALL_OBJECT_VOLUME_THRESHOLD:
                _small_indices.append(idx)
                print(f"  [Step0] '{e['category']}'({e['uid']}): 体積={_vol:.3f}m^3 -> 小さいオブジェクト"
                      f"(閾値{SMALL_OBJECT_VOLUME_THRESHOLD}m^3未満)")
            else:
                _large_indices.append(idx)
                print(f"  [Step0] '{e['category']}'({e['uid']}): 体積={_vol:.3f}m^3 -> 大きいオブジェクト")

        print(f"\n[Step0] 大きいオブジェクト: {len(_large_indices)}件 / 小さいオブジェクト: {len(_small_indices)}件"
              f" (うちカテゴリ名で強制的に小さい扱いにしたもの: {_n_force_small}件)")


        # ============================================================
        # Step1+2: 小さいオブジェクトの設置先を探し、親子関係を記録する
        #     (積み重ねは許可しない: 候補は「大きいオブジェクトの面」と「床」のみ。
        #      小さいオブジェクト同士は互いの候補にならない)
        # ============================================================
        _floor_plane_mesh = trimesh.creation.box(extents=[FLOOR_PLANE_SIZE, 0.01, FLOOR_PLANE_SIZE])
        _floor_plane_mesh.apply_translation([0.0, -0.005, 0.0])

        _landing_surfaces = []
        for idx in _large_indices:
            for part_mesh in _collider_lookup[idx]:
                for group_mesh in _extract_upward_face_groups(part_mesh):
                    desc = _surface_descriptor(group_mesh, idx)
                    if desc is not None:
                        _landing_surfaces.append(desc)

        _floor_surfaces = []
        for group_mesh in _extract_upward_face_groups(_floor_plane_mesh):
            desc = _surface_descriptor(group_mesh, None)
            if desc is not None:
                _floor_surfaces.append(desc)

        _all_candidate_surfaces = _landing_surfaces + _floor_surfaces
        print(f"[Step1] 設置先の候補面: 大きいオブジェクト由来 {len(_landing_surfaces)}枚 + 床 {len(_floor_surfaces)}枚")

        _parent_of = {}        # child_idx -> parent_idx(Noneなら親なし=床 or 見つからず)
        _relative_offset = {}  # child_idx -> ワールド座標での親からの相対位置ベクトル

        for idx in _small_indices:
            e = _placement_entries[idx]
            size_whl = e["size_whl"]
            world_rotation = e["world_rotation"]
            world_position = e["world_position"]
            height = float(size_whl[1])
            current_bottom_y = float(world_position[1]) - height / 2.0
            current_xz = np.array([world_position[0], world_position[2]])
            corner_offsets_xz = _obb_footprint_corners_xz(size_whl, world_rotation)

            best = None  # (surface, score, landed_xz, landed_y)
            for s in _all_candidate_surfaces:
                cx, cz = s["centroid_xz"]
                dist_to_current = float(np.hypot(cx - current_xz[0], cz - current_xz[1]))

                # まず現在位置のままで収まるかを確認し、収まらなければ「収まる最小限の
                # 移動先」を探す(探索半径内のみ)
                corners_now = current_xz + corner_offsets_xz
                if _footprint_overlap_ratio(s["polygon"], corners_now) >= LANDING_MIN_OVERLAP_RATIO:
                    landed_xz = current_xz
                else:
                    if dist_to_current > CHILD_SEARCH_RADIUS:
                        continue
                    landed_xz = _search_nearby_fit(
                        current_xz, corner_offsets_xz, s, LANDING_MIN_OVERLAP_RATIO, CHILD_SEARCH_RADIUS
                    )
                    if landed_xz is None:
                        continue

                landing_y = _plane_height(s, landed_xz[0], landed_xz[1])
                height_diff = abs(landing_y - current_bottom_y)
                if height_diff > CHILD_MAX_HEIGHT_DIFF:
                    continue

                move_dist = float(np.hypot(landed_xz[0] - current_xz[0], landed_xz[1] - current_xz[1]))
                score = CHILD_HEIGHT_WEIGHT * height_diff + CHILD_DISTANCE_WEIGHT * move_dist
                if best is None or score < best[1]:
                    best = (s, score, landed_xz, landing_y)

            if best is None:
                _parent_of[idx] = None
                print(f"  [Step1/2] '{e['category']}'({e['uid']}): 収まる設置先が見つかりませんでした"
                      f"(位置はそのまま、親なし)")
                continue

            chosen_surface, _score, landed_xz, landing_y = best
            new_y = landing_y + height / 2.0
            new_position = np.array([landed_xz[0], new_y, landed_xz[1]])
            delta = new_position - world_position
            _move_dist = float(np.linalg.norm(delta[[0, 2]]))
            e["world_position"] = new_position
            for m in _collider_lookup[idx]:
                m.apply_translation(delta)

            if chosen_surface["entry_index"] is not None:
                parent_idx = chosen_surface["entry_index"]
                _parent_of[idx] = parent_idx
                _relative_offset[idx] = new_position - _placement_entries[parent_idx]["world_position"]
                _parent_e = _placement_entries[parent_idx]
                print(f"  [Step1/2] '{e['category']}'({e['uid']}): "
                      f"'{_parent_e['category']}'({_parent_e['uid']})の上に設置(親子付け)"
                      f" / 水平移動={_move_dist:.3f}m, 高さ={landing_y:.3f}m")
            else:
                _parent_of[idx] = None  # 床に着地(親なし)
                print(f"  [Step1/2] '{e['category']}'({e['uid']}): 床の上に設置(親なし)"
                      f" / 水平移動={_move_dist:.3f}m, 高さ={landing_y:.3f}m")

        _children_of = {}
        for child_idx, parent_idx in _parent_of.items():
            if parent_idx is not None:
                _children_of.setdefault(parent_idx, []).append(child_idx)

        _child_indices_set = {i for i, p in _parent_of.items() if p is not None}
        _parent_indices = [
            i for i in range(len(_placement_entries))
            if i in _collider_lookup and i not in _child_indices_set
        ]

        print(f"[Step2] 親子付けされた小物: {len(_child_indices_set)}件"
              f" / 独立した(親を持たない)オブジェクト: {len(_parent_indices) - len(_large_indices)}件"
              f" / 大きいオブジェクト: {len(_large_indices)}件")


        # ============================================================
        # Step3: 「親」(子でない全オブジェクト)を、中心からの真下レイで接地する。
        #     動いた親を持つ子は、記録した相対位置を保って追従させる。
        # ============================================================
        def _move_entry(idx, delta):
            _placement_entries[idx]["world_position"] = _placement_entries[idx]["world_position"] + delta
            for _m in _collider_lookup[idx]:
                _m.apply_translation(delta)


        def _reattach_children(parent_idx):
            for child_idx in _children_of.get(parent_idx, []):
                new_pos = _placement_entries[parent_idx]["world_position"] + _relative_offset[child_idx]
                cdelta = new_pos - _placement_entries[child_idx]["world_position"]
                if np.linalg.norm(cdelta) > 1e-9:
                    _move_entry(child_idx, cdelta)


        _n_snapped = 0
        _n_already_on_ground = 0

        for idx in _parent_indices:
            e = _placement_entries[idx]
            world_position = e["world_position"]
            size_whl = e["size_whl"]
            world_rotation = e["world_rotation"]
            height = float(size_whl[1])
            current_bottom_y = float(world_position[1]) - height / 2.0

            if abs(current_bottom_y) <= GROUND_CONTACT_TOLERANCE:
                _n_already_on_ground += 1
                print(f"  [Step3] '{e['category']}'({e['uid']}): 既に接地済み(底面Y={current_bottom_y:.3f})")
                continue

            # 底面の中心から真下にレイを飛ばし、当たった場所をそのまま採用する
            # (面積のチェックはしない)。自分自身のコライダーは_support_meshesに含めて
            # いないので、底面ちょうどから出しても自己衝突の心配はない。
            _support_meshes = [_floor_plane_mesh]
            if SUPPORT_MUST_BE_BIGGER_FACTOR is not None:
                _self_volume = _obb_volume(size_whl)
                for _other_idx in _parent_indices:
                    if _other_idx == idx:
                        continue
                    _other_e = _placement_entries[_other_idx]
                    if _obb_volume(_other_e["size_whl"]) >= _self_volume * SUPPORT_MUST_BE_BIGGER_FACTOR:
                        _support_meshes.extend(_collider_lookup[_other_idx])
            _support_mesh = trimesh.util.concatenate(_support_meshes)

            ray_origin = np.array([[world_position[0], current_bottom_y, world_position[2]]])
            ray_direction = np.array([[0.0, -1.0, 0.0]])
            locations, _idx_ray, _idx_tri = _support_mesh.ray.intersects_location(
                ray_origin, ray_direction, multiple_hits=True
            )
            if len(locations) == 0:
                support_y = 0.0
                _landing_note = "床(レイが何にも当たらなかったためフォールバック)"
            else:
                support_y = float(np.max(locations[:, 1]))
                _landing_note = "レイが当たった場所"

            new_y = support_y + height / 2.0
            delta = np.array([0.0, new_y - world_position[1], 0.0])
            if np.linalg.norm(delta) > 1e-9:
                _n_children = len(_children_of.get(idx, []))
                _move_entry(idx, delta)
                _n_snapped += 1
                _reattach_children(idx)
                print(f"  [Step3] '{e['category']}'({e['uid']}): "
                      f"底面Y {current_bottom_y:.3f} -> {support_y:.3f} に接地({_landing_note})"
                      f"({delta[1]:+.3f}m" + (f", 子{_n_children}件も追従)" if _n_children else ")"))
            else:
                print(f"  [Step3] '{e['category']}'({e['uid']}): 接地判定は行われましたが、ほぼ移動不要でした"
                      f"({_landing_note})")

        print(f"\n[Step3] {_n_snapped}件を接地 / {_n_already_on_ground}件は既に接地済み")


        # ============================================================
        # Step4: 「親」どうしのめり込みを解消する(子は一切参加しない)。
        #     解消後、子は親の新しい位置 + 相対位置へ再配置する。
        # ============================================================
        # 壁メッシュ(Section 12.7で作成)があれば、動かない障害物として一緒に登録する。
        # これにより、家具どうしのめり込みだけでなく、家具が壁にめり込んでいる場合も
        # 壁の外(部屋の内側)へ押し出されるようになる。
        _WALL_OBJECT_NAME = "WALL"
        _wall_available = ("wall_mesh" in locals() or "wall_mesh" in globals()) and wall_mesh is not None and len(wall_mesh.faces) > 0

        _manager = trimesh.collision.CollisionManager()
        for idx in _parent_indices:
            for pi, m in enumerate(_collider_lookup[idx]):
                _manager.add_object(f"{idx}:{pi}", m)

        if _wall_available:
            _manager.add_object(_WALL_OBJECT_NAME, wall_mesh)
            print("[Step4] 壁メッシュも障害物として追加しました(家具は壁からも押し出されます)。")
        else:
            print("[Step4] 壁メッシュが見つからないため、家具どうしのめり込み解消のみ行います"
                  "(Section 12.7を実行すると壁からの押し出しも有効になります)。")

        _disp = {idx: np.zeros(3) for idx in _parent_indices}
        _initial_colliding_pairs = None
        _initial_wall_contacts = None
        _iterations_used = 0

        for _iteration in range(OVERLAP_MAX_ITERATIONS):
            _iterations_used = _iteration + 1
            _is_collision, _names, _data = _manager.in_collision_internal(return_names=True, return_data=True)

            _pair_depth = {}     # (idx_a, idx_b) -> depth (家具どうし)
            _wall_contacts = {}  # idx -> (depth, contact_point) (家具 vs 壁。深い方を残す)

            for _contact in _data:
                _name_a, _name_b = tuple(_contact.names)
                if _name_a == _WALL_OBJECT_NAME or _name_b == _WALL_OBJECT_NAME:
                    _other_name = _name_b if _name_a == _WALL_OBJECT_NAME else _name_a
                    _idx = int(_other_name.split(":")[0])
                    _prev = _wall_contacts.get(_idx)
                    if _prev is None or _contact.depth > _prev[0]:
                        _wall_contacts[_idx] = (_contact.depth, _contact.point)
                    continue
                _idx_a = int(_name_a.split(":")[0])
                _idx_b = int(_name_b.split(":")[0])
                if _idx_a == _idx_b:
                    continue
                _key = tuple(sorted((_idx_a, _idx_b)))
                _pair_depth[_key] = max(_pair_depth.get(_key, 0.0), _contact.depth)

            if _iteration == 0:
                _initial_colliding_pairs = set(_pair_depth.keys())
                _initial_wall_contacts = set(_wall_contacts.keys())

            if not _pair_depth and not _wall_contacts:
                _iterations_used = _iteration
                break

            _all_depths = list(_pair_depth.values()) + [d for d, _p in _wall_contacts.values()]
            _max_depth = max(_all_depths) if _all_depths else 0.0
            if _max_depth < OVERLAP_CONVERGENCE_TOLERANCE:
                _iterations_used = _iteration
                break

            _delta = {idx: np.zeros(3) for idx in _parent_indices}

            # --- 家具どうしの押し出し(従来通り、深さの半分ずつ互いに押し合う) ---
            for (_idx_a, _idx_b), _depth in _pair_depth.items():
                _pos_a = _placement_entries[_idx_a]["world_position"] + _disp[_idx_a]
                _pos_b = _placement_entries[_idx_b]["world_position"] + _disp[_idx_b]
                _dir_xz = np.array([_pos_b[0] - _pos_a[0], 0.0, _pos_b[2] - _pos_a[2]])
                _dir_norm = np.linalg.norm(_dir_xz)
                if _dir_norm < 1e-6:
                    _dir_xz = np.array([1.0, 0.0, 0.0])
                    _dir_norm = 1.0
                _dir_xz = _dir_xz / _dir_norm
                _push = _depth * OVERLAP_RELAXATION * 0.5
                _delta[_idx_a] -= _dir_xz * _push
                _delta[_idx_b] += _dir_xz * _push

            # --- 壁からの押し出し: 壁は動かないので、家具側だけを壁面の法線方向(部屋の
            #     内側)へ深さの分だけフルに押す。どの壁パネルに当たったかは、接触点に
            #     最も近い壁メッシュの面を探してその法線を使う。 ---
            for _idx, (_depth, _point) in _wall_contacts.items():
                _closest_pts, _dists, _face_idxs = wall_mesh.nearest.on_surface([_point])
                _face_idx = int(_face_idxs[0])
                _wall_normal = wall_mesh.face_normals[_face_idx]
                _wall_normal_xz = np.array([_wall_normal[0], 0.0, _wall_normal[2]])
                _wn_norm = np.linalg.norm(_wall_normal_xz)
                if _wn_norm < 1e-6:
                    continue
                _wall_normal_xz = _wall_normal_xz / _wn_norm
                _push = _depth * OVERLAP_RELAXATION
                _delta[_idx] += _wall_normal_xz * _push

            for idx in _parent_indices:
                _disp[idx] += _delta[idx]
                for _pi in range(len(_collider_lookup[idx])):
                    _manager.set_transform(f"{idx}:{_pi}", trimesh.transformations.translation_matrix(_disp[idx]))
                # 壁(_WALL_OBJECT_NAME)は動かさないのでtransformの更新は不要

        _final_is_collision, _final_names, _final_data = _manager.in_collision_internal(return_names=True, return_data=True)
        _remaining_pairs = set()
        _remaining_wall_contacts = set()
        for _contact in _final_data:
            _name_a, _name_b = tuple(_contact.names)
            if _name_a == _WALL_OBJECT_NAME or _name_b == _WALL_OBJECT_NAME:
                if _contact.depth >= OVERLAP_CONVERGENCE_TOLERANCE:
                    _other_name = _name_b if _name_a == _WALL_OBJECT_NAME else _name_a
                    _remaining_wall_contacts.add(int(_other_name.split(":")[0]))
                continue
            _idx_a = int(_name_a.split(":")[0])
            _idx_b = int(_name_b.split(":")[0])
            if _idx_a != _idx_b and _contact.depth >= OVERLAP_CONVERGENCE_TOLERANCE:
                _remaining_pairs.add(tuple(sorted((_idx_a, _idx_b))))

        _n_moved = sum(1 for v in _disp.values() if np.linalg.norm(v) > 1e-6)
        print(f"\n[Step4] もともと衝突していた親どうしのペア: {len(_initial_colliding_pairs or set())}組"
              f" / 壁にめり込んでいた家具: {len(_initial_wall_contacts or set())}件")
        print(f"反復回数: {_iterations_used} / {OVERLAP_MAX_ITERATIONS}")
        print(f"位置を補正した親: {_n_moved}件")
        if _remaining_pairs or _remaining_wall_contacts:
            if _remaining_pairs:
                print(f"  [WARN] {len(_remaining_pairs)}組は反復回数の上限までに解消しきれませんでした:")
                for _idx_a, _idx_b in _remaining_pairs:
                    _e_a, _e_b = _placement_entries[_idx_a], _placement_entries[_idx_b]
                    print(f"    - '{_e_a['category']}'({_e_a['uid']}) <-> '{_e_b['category']}'({_e_b['uid']})")
            if _remaining_wall_contacts:
                print(f"  [WARN] {len(_remaining_wall_contacts)}件は壁とのめり込みを解消しきれませんでした:")
                for _idx in _remaining_wall_contacts:
                    _e = _placement_entries[_idx]
                    print(f"    - '{_e['category']}'({_e['uid']})")
        elif _initial_colliding_pairs or _initial_wall_contacts:
            print("すべての衝突(壁との接触も含む)を解消しました。")
        else:
            print("もともと衝突している親ペア・壁との接触はありませんでした。")

        for idx, d in _disp.items():
            if np.linalg.norm(d) < 1e-9:
                continue
            _move_entry(idx, d)
            _reattach_children(idx)


        # ============================================================
        # 補正後の位置で、実際の家具メッシュを再配置してエクスポートする
        #     (色合わせはSection 12.5と同じロジック)
        # ============================================================

        def _extract_target_color_clusters(crop_path, mask_path, box, n_clusters=RECOLOR_N_COLOR_CLUSTERS):
            img = Image.open(crop_path).convert("RGB")
            img_arr = np.array(img).astype(np.float64)
            pixels = None
            if mask_path and os.path.exists(mask_path) and box is not None:
                x1, y1, x2, y2 = [int(round(v)) for v in box]
                mask_full = Image.open(mask_path).convert("L")
                mask_crop = mask_full.crop((x1, y1, x2, y2))
                mask_arr = np.array(mask_crop) > 127
                if mask_arr.shape[:2] == img_arr.shape[:2] and mask_arr.any():
                    pixels = img_arr[mask_arr]
            if pixels is None or len(pixels) == 0:
                pixels = img_arr.reshape(-1, 3)
            k = min(n_clusters, max(1, len(pixels) // 30))
            if k <= 1:
                return [(np.median(pixels, axis=0), 1.0)]
            rng = np.random.default_rng(0)
            sample = pixels if len(pixels) <= 5000 else pixels[rng.choice(len(pixels), 5000, replace=False)]
            centroids01, labels = kmeans2(sample / 255.0, k, seed=0, minit="++")
            centroids = np.clip(centroids01 * 255.0, 0, 255)
            counts = np.bincount(labels, minlength=k)
            weights = counts / max(1, counts.sum())
            order = np.argsort(-weights)
            return [(centroids[i], float(weights[i])) for i in order if weights[i] > 0]


        def _match_vertex_colors_to_targets(orig_colors_u8, target_clusters):
            colors01 = orig_colors_u8[:, :3].astype(np.float64) / 255.0
            n = len(colors01)
            target_rgbs = np.array([c for c, _w in target_clusters])
            k = min(len(target_clusters), max(1, n // 30))
            if k <= 1 or n < 30:
                return np.tile(target_rgbs[0], (n, 1))
            sample_n = min(n, 4000)
            rng = np.random.default_rng(0)
            sample_idx = rng.choice(n, sample_n, replace=False) if n > sample_n else np.arange(n)
            mesh_centroids, _ = kmeans2(colors01[sample_idx], k, seed=0, minit="++")
            dists = np.linalg.norm(colors01[:, None, :] - mesh_centroids[None, :, :], axis=2)
            vertex_cluster = np.argmin(dists, axis=1)
            mesh_v = mcolors.rgb_to_hsv(np.clip(mesh_centroids, 0.0, 1.0))[:, 2]
            mesh_order = np.argsort(mesh_v)
            target_v = mcolors.rgb_to_hsv(np.clip(target_rgbs / 255.0, 0.0, 1.0))[:, 2]
            target_order = np.argsort(target_v)
            cluster_to_target = np.zeros(k, dtype=int)
            for rank, mesh_ci in enumerate(mesh_order):
                t_rank = int(round(rank / max(1, k - 1) * (len(target_order) - 1))) if k > 1 else 0
                cluster_to_target[mesh_ci] = target_order[t_rank]
            return target_rgbs[cluster_to_target[vertex_cluster]]


        def _hue_shift_vertex_colors(vertex_colors_rgba_u8, target_rgb_per_vertex, hue_strength, saturation_strength,
                                      min_target_saturation=RECOLOR_MIN_TARGET_SATURATION):
            colors01 = vertex_colors_rgba_u8.astype(np.float64) / 255.0
            rgb, alpha = colors01[:, :3], colors01[:, 3]
            hsv = mcolors.rgb_to_hsv(rgb)
            h, s, v = hsv[:, 0], hsv[:, 1], hsv[:, 2]
            target_rgb01 = np.clip(np.asarray(target_rgb_per_vertex, dtype=np.float64) / 255.0, 0.0, 1.0)
            if target_rgb01.ndim == 1:
                target_rgb01 = np.tile(target_rgb01, (len(rgb), 1))
            target_hsv = mcolors.rgb_to_hsv(target_rgb01)
            target_h, target_s = target_hsv[:, 0], target_hsv[:, 1]
            effective_hue_strength = hue_strength * np.clip(target_s / max(min_target_saturation, 1e-6), 0.0, 1.0)
            delta_h = (target_h - h + 0.5) % 1.0 - 0.5
            new_h = (h + delta_h * effective_hue_strength) % 1.0
            new_s = np.clip(s + (target_s - s) * saturation_strength, 0.0, 1.0)
            new_rgb = mcolors.hsv_to_rgb(np.stack([new_h, new_s, v], axis=1))
            new_colors = np.clip(new_rgb * 255.0, 0, 255).astype(np.uint8)
            new_alpha = np.clip(alpha * 255.0, 0, 255).astype(np.uint8)
            return np.column_stack([new_colors, new_alpha])


        if os.path.isdir(COLLISION_DOWNLOAD_DIR):
            shutil.rmtree(COLLISION_DOWNLOAD_DIR, ignore_errors=True)
        os.makedirs(COLLISION_DOWNLOAD_DIR, exist_ok=True)

        _fixed_uid_to_path = _download_hssd_objects_to(
            sorted({e["uid"] for e in _placement_entries}), COLLISION_DOWNLOAD_DIR
        )

        fixed_scene = trimesh.Scene()
        _n_fixed_placed = 0
        for e in _placement_entries:
            local_path = _fixed_uid_to_path.get(e["uid"])
            if not local_path:
                print(f"  [WARN] {e['uid']} のダウンロードに失敗しているためスキップ")
                continue
            try:
                mesh = trimesh.load(local_path, force="mesh")

                if RECOLOR_TO_MATCH_PHOTO:
                    if "target_color_clusters" not in e:
                        _crop_path = os.path.join(output_dir, e["file"]) if e.get("file") else None
                        _mask_path = os.path.join(output_dir, e["mask_file"]) if e.get("mask_file") else None
                        if _crop_path and os.path.exists(_crop_path):
                            e["target_color_clusters"] = _extract_target_color_clusters(_crop_path, _mask_path, e.get("box"))
                        else:
                            e["target_color_clusters"] = None
                    if e.get("target_color_clusters"):
                        _current_colors = mesh.visual.to_color().vertex_colors.astype(np.uint8)
                        _per_vertex_target_rgb = _match_vertex_colors_to_targets(_current_colors, e["target_color_clusters"])
                        _new_colors = _hue_shift_vertex_colors(
                            _current_colors, _per_vertex_target_rgb, RECOLOR_HUE_STRENGTH, RECOLOR_SATURATION_STRENGTH,
                        )
                        mesh.visual = trimesh.visual.color.ColorVisuals(mesh=mesh, vertex_colors=_new_colors)

                local_center = mesh.bounds.mean(axis=0)
                extents = mesh.bounds[1] - mesh.bounds[0]
                scale_xyz = _scale_for_extra_rotation(extents, e["size_whl"], e.get("mesh_extra_yaw_deg", 0))
                transform = _build_placement_transform(
                    local_center, scale_xyz, e["world_rotation"], e["world_position"]
                )
                placed = mesh.copy()
                placed.apply_transform(transform)
                fixed_scene.add_geometry(placed, node_name=f"{e['category']}_{_n_fixed_placed}")
                _n_fixed_placed += 1
            except Exception as ex:
                print(f"  [WARN] 再配置失敗: {e['uid']} ({e['category']}): {ex!r}")

        shutil.rmtree(COLLISION_DOWNLOAD_DIR, ignore_errors=True)
        print(f"\n補正後の位置で家具メッシュを再配置しました: {_n_fixed_placed} / {len(_placement_entries)}件")

        if _n_fixed_placed > 0:
            fixed_scene.export(FIXED_SCENE_OUTPUT_PATH)
            print("saved:", FIXED_SCENE_OUTPUT_PATH)
        else:
            print("再配置できた家具がありませんでした。")


        fixed_collider_scene = trimesh.Scene()
        for r in collision_results:
            for i, part_mesh in enumerate(r["collider_meshes"]):
                vis_mesh = part_mesh.copy()
                vis_mesh.visual.face_colors = _PART_COLOR_PALETTE[i % len(_PART_COLOR_PALETTE)]
                fixed_collider_scene.add_geometry(vis_mesh, node_name=f"collider_{r['category']}_{r['uid']}_{i}")

        if len(fixed_collider_scene.geometry) > 0:
            fixed_collider_scene.export(FIXED_COLLIDER_SCENE_OUTPUT_PATH)
            print("saved:", FIXED_COLLIDER_SCENE_OUTPUT_PATH)


        progress({"percent": _PCT["unity_export"], "label": "Unity向け出力を書き出し中"})

        # ---- unity_export (元ノートブック cell 60) ----
        import trimesh.transformations as _tf_unity

        # ==========================================
        UNITY_VISUAL_SCENE_PATH = os.path.join(WORK_DIR, "unity_visual.glb")
        UNITY_COLLISION_SCENE_PATH = os.path.join(WORK_DIR, "unity_collision.glb")
        UNITY_MANIFEST_PATH = os.path.join(WORK_DIR, "unity_manifest.json")
        # ==========================================

        # --- 家具ごとに安定したノード名を決める(可視化用・当たり判定用の両方で同じ名前を
        #     使うので、Unity側でスクリプトから対応付けやすい)。 ---
        def _unity_node_name(idx, e):
            _safe_category = "".join(c if c.isalnum() else "_" for c in e["category"])
            return f"{idx:03d}_{_safe_category}_{e['uid'][:8]}"


        if os.path.isdir(COLLISION_DOWNLOAD_DIR):
            shutil.rmtree(COLLISION_DOWNLOAD_DIR, ignore_errors=True)
        os.makedirs(COLLISION_DOWNLOAD_DIR, exist_ok=True)

        _unity_uid_to_path = _download_hssd_objects_to(
            sorted({e["uid"] for e in _placement_entries}), COLLISION_DOWNLOAD_DIR
        )

        unity_visual_scene = trimesh.Scene()
        unity_collision_scene = trimesh.Scene()
        _manifest_entries = []
        _n_unity_placed = 0

        for idx, e in enumerate(_placement_entries):
            local_path = _unity_uid_to_path.get(e["uid"])
            node_name = _unity_node_name(idx, e)

            # --- ワールド変換行列(位置・回転・スケール)を組み立てる。Node側にそのまま
            #     Transformとして持たせるので、メッシュ自体には焼き込まない。 ---
            world_rotation = e["world_rotation"]
            world_position = e["world_position"]

            if local_path:
                try:
                    mesh = trimesh.load(local_path, force="mesh")

                    if RECOLOR_TO_MATCH_PHOTO:
                        if "target_color_clusters" not in e:
                            _crop_path = os.path.join(output_dir, e["file"]) if e.get("file") else None
                            _mask_path = os.path.join(output_dir, e["mask_file"]) if e.get("mask_file") else None
                            if _crop_path and os.path.exists(_crop_path):
                                e["target_color_clusters"] = _extract_target_color_clusters(
                                    _crop_path, _mask_path, e.get("box")
                                )
                            else:
                                e["target_color_clusters"] = None
                        if e.get("target_color_clusters"):
                            _current_colors = mesh.visual.to_color().vertex_colors.astype(np.uint8)
                            _per_vertex_target_rgb = _match_vertex_colors_to_targets(
                                _current_colors, e["target_color_clusters"]
                            )
                            _new_colors = _hue_shift_vertex_colors(
                                _current_colors, _per_vertex_target_rgb, RECOLOR_HUE_STRENGTH, RECOLOR_SATURATION_STRENGTH,
                            )
                            mesh.visual = trimesh.visual.color.ColorVisuals(mesh=mesh, vertex_colors=_new_colors)

                    local_center = mesh.bounds.mean(axis=0)
                    extents = mesh.bounds[1] - mesh.bounds[0]
                    scale_xyz = _scale_for_extra_rotation(extents, e["size_whl"], e.get("mesh_extra_yaw_deg", 0))
                    full_transform = _build_placement_transform(local_center, scale_xyz, world_rotation, world_position)

                    # --- 可視化用メッシュ: 頂点は「原点中心・スケール適用済み・回転前」の
                    #     ローカル空間のままにしておき、Node側に(回転+平行移動)だけを
                    #     Transformとして持たせる(スケールは形状の一部としてメッシュに
                    #     残す。glTFのNodeスケールにも本来乗せられるが、コライダー側との
                    #     対応を単純にするため、ここではメッシュ側に含めている)。 ---
                    mesh_local = mesh.copy()
                    mesh_local.apply_translation(-local_center)
                    mesh_local.apply_scale(scale_xyz)

                    node_transform = np.eye(4)
                    node_transform[:3, :3] = world_rotation
                    node_transform[:3, 3] = world_position

                    unity_visual_scene.add_geometry(mesh_local, node_name=node_name, transform=node_transform)
                    _n_unity_placed += 1
                except Exception as ex:
                    print(f"  [WARN] 可視化メッシュの再配置に失敗: {e['uid']} ({e['category']}): {ex!r}")
                    full_transform = None
            else:
                print(f"  [WARN] {e['uid']} のダウンロードに失敗しているため可視化メッシュをスキップ")
                full_transform = None

            # --- 当たり判定メッシュ: collision_resultsのコライダー(ワールド空間で
            #     生成済み)を、同じnode_transformの逆行列でローカル空間へ戻し、
            #     可視化メッシュと全く同じ名前・Transformの下にぶら下げる。 ---
            _colliders = _collider_lookup.get(idx)
            if _colliders:
                node_transform = np.eye(4)
                node_transform[:3, :3] = world_rotation
                node_transform[:3, 3] = world_position
                _inv_transform = np.linalg.inv(node_transform)

                # 親ノード(家具1つぶん)を、可視化用と同じTransformで作る
                unity_collision_scene.graph.update(frame_to=node_name, matrix=node_transform)
                for _pi, _part_mesh in enumerate(_colliders):
                    _local_part = _part_mesh.copy()
                    _local_part.apply_transform(_inv_transform)
                    unity_collision_scene.add_geometry(
                        _local_part, node_name=f"{node_name}_col{_pi}", parent_node_name=node_name
                    )

            _quat_wxyz = _tf_unity.quaternion_from_matrix(
                np.vstack([np.hstack([world_rotation, [[0], [0], [0]]]), [0, 0, 0, 1]])
            )
            _manifest_entries.append({
                "index": idx,
                "node_name": node_name,
                "category": e["category"],
                "uid": e["uid"],
                "parent_index": (_parent_of.get(idx) if ("_parent_of" in locals() or "_parent_of" in globals()) else None),  # Section 13.5の親子付け
                "size_whl": [float(v) for v in e["size_whl"]],
                "world_position": [float(v) for v in world_position],
                "world_rotation_quaternion_wxyz": [float(v) for v in _quat_wxyz],
                "has_visual_mesh": bool(local_path),
                "has_collision_mesh": bool(_colliders),
                "n_collision_parts": len(_colliders) if _colliders else 0,
            })

        shutil.rmtree(COLLISION_DOWNLOAD_DIR, ignore_errors=True)
        print(f"[unity export] 可視化メッシュを配置しました: {_n_unity_placed} / {len(_placement_entries)}件")

        unity_visual_scene.export(UNITY_VISUAL_SCENE_PATH)
        print("saved:", UNITY_VISUAL_SCENE_PATH)

        unity_collision_scene.export(UNITY_COLLISION_SCENE_PATH)
        print("saved:", UNITY_COLLISION_SCENE_PATH)

        with open(UNITY_MANIFEST_PATH, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "entries": _manifest_entries,
                    "wall_glb": os.path.basename(WALL_SCENE_OUTPUT_PATH) if ("WALL_SCENE_OUTPUT_PATH" in locals() or "WALL_SCENE_OUTPUT_PATH" in globals()) else None,
                    "visual_glb": os.path.basename(UNITY_VISUAL_SCENE_PATH),
                    "collision_glb": os.path.basename(UNITY_COLLISION_SCENE_PATH),
                },
                f, ensure_ascii=False, indent=2,
            )
        print("saved:", UNITY_MANIFEST_PATH)


        # ==================== 出力をbase64にまとめて返す ====================
        def _b64(path):
            with open(path, "rb") as f:
                return base64.b64encode(f.read()).decode("ascii")

        with open(UNITY_MANIFEST_PATH, encoding="utf-8") as f:
            manifest = json.load(f)

        report({"percent": 100, "label": "完了"})

        return {
            "walls_glb_base64": _b64(WALL_SCENE_OUTPUT_PATH),
            "visual_glb_base64": _b64(UNITY_VISUAL_SCENE_PATH),
            "collision_glb_base64": _b64(UNITY_COLLISION_SCENE_PATH),
            "manifest": manifest,
        }
    finally:
        # 一時ファイルを掃除する(重み・DBなどはBASE_DIR配下の別フォルダなので消えない)。
        # 掃除する前に、必ず「これから消すWORK_DIRの外」へ移動しておく。
        # (os.chdir(WORK_DIR)したまま、そのWORK_DIRをrmtreeすると、プロセスの
        #  カレントディレクトリが「存在しない場所」になり、以降このプロセス上で
        #  シェルコマンド(!pip installなど)を実行すると
        #  "getcwd() failed: No such file or directory" になる)
        try:
            os.chdir(BASE_DIR)
        except Exception:
            pass
        shutil.rmtree(WORK_DIR, ignore_errors=True)
        gc.collect()
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
