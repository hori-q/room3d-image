# 専用Dockerイメージの作り方と使い方

このフォルダのファイルで、RunPod用の専用イメージ(`room3d-wilddet3d`)を作ります。

| ファイル | 役割 |
|---|---|
| `Dockerfile` | イメージの設計図(torch 2.5.1、pipパッケージ、CUDA版vis4d_cuda_ops、WildDet3D、JupyterLab) |
| `start.sh` | Pod起動時にJupyterLabを立ち上げるスクリプト |
| `build-and-push.yml` | (任意)GitHub Actionsでビルドする場合の設定 |
| `../3dscenerestruction_walls_only_image.ipynb` | イメージ用のノートブック(pip・ビルドを省いた版) |

> **未検証です。** このDockerfileは、GPUもDockerも無い環境で作ったため、実際にビルドして動かしてはいません。
> 初回は、ビルド中や起動時にエラーが出る可能性があります。エラーが出たら、そのメッセージを教えてください。

## 0. 準備

1. **Docker Hubのアクセストークンを作る**
   Docker Hub → 右上のアカウント → Account settings → Personal access tokens → Generate new token
   (権限は **Read & Write**)。表示されたトークンは、一度しか見られないのでメモしておく。
2. **ビルドする場所を決める**

| 場所 | 条件 |
|---|---|
| **自分のパソコン**(Docker Desktop) | Windows(WSL2)またはLinux/Intel Mac。**ディスクの空き50GB以上**。Pushに使う上り回線が速いこと |
| **Apple Silicon(M1/M2/M3…)のMac** | **おすすめしません**。amd64向けのCUDAビルドをエミュレーションで行うと、非常に遅く、失敗しやすいです。下の「GitHub Actionsでビルドする」を使ってください |
| **GitHub Actions** | GitHubアカウントが必要。ローカルのDockerは不要 |

## 1. 自分のパソコンでビルドする場合

`Dockerfile` と `start.sh` を、同じフォルダ(例: `room3d-image/`)に置き、そのフォルダで実行します。
`ユーザー名` は、Docker Hubのユーザー名に置き換えてください。

```bash
# 1) ログイン(パスワードを聞かれたら、アクセストークンを貼り付ける)
docker login -u ユーザー名

# 2) ビルド(30〜90分が目安。GPUが無くても、CUDA版をビルドする設定になっています)
docker build -t ユーザー名/room3d-wilddet3d:1.0 .

# 3) Push(イメージは10GB前後。回線によって数十分〜数時間)
docker push ユーザー名/room3d-wilddet3d:1.0
```

### ビルド中に確認するポイント
ビルドのログに、次のような行が出れば、CUDA版として正しく作られています。

```
埋め込まれたSASSの世代: ['75', '80', '86', '89']
torch 2.5.1... | CUDA 12.4 | transformers 5.15.1
```

- 「`vis4d_cuda_ops がCUDA版になっていません`」で失敗した場合 → 教えてください(Dockerfileの修正が必要です)。
- 「`torch が入れ替わっています`」で失敗した場合 → どのパッケージが入れ替えたかを調べる必要があります。教えてください。

## 2. GitHub Actionsでビルドする場合

1. GitHubで、新しいリポジトリ(非公開でよい)を作る。
2. リポジトリ直下に `Dockerfile` と `start.sh` を、`.github/workflows/` に `build-and-push.yml` を置いてPushする。
3. リポジトリの Settings → Secrets and variables → Actions で、次の2つを登録する。
   - `DOCKERHUB_USERNAME`: Docker Hubのユーザー名
   - `DOCKERHUB_TOKEN`: 0.で作ったアクセストークン
4. Actionsタブ → `build-and-push` → Run workflow。

> イメージが大きいため、ランナーの空き容量が不足する可能性があります(ファイルの中で、不要なファイルを削除する手順を入れています)。
> それでも足りない場合は、教えてください。

### 失敗したとき、最初からやり直しになるか
ビルドは4つの段階(`pkgs` → `ops` → `wd3d` → `final`)に分けてあり、**成功した段階ごとに、キャッシュをDocker Hubに保存**します
(同じリポジトリの `buildcache` タグ。公開リポジトリなので、容量は数十GB使っても無料枠の制限はありません)。

| 失敗した場所 | 再実行したとき |
|---|---|
| 段階1(pkgs) | 段階1の最初からやり直し(pipパッケージの数分〜十数分) |
| 段階2(ops) | 段階1はキャッシュから再利用され、**段階2(CUDAビルド)だけやり直し** |
| 段階3(wd3d) | 段階1・2はキャッシュから再利用され、段階3だけやり直し |
| 段階4(final) | 段階1〜3はキャッシュから再利用され、段階4だけやり直し |

- 失敗した段階の**途中まで進んだ分は、保存されません**(キャッシュは、段階が成功したときにだけ保存されます)。
- キャッシュの取得にも、数分かかります。
- Dockerfileの**前のほうを変更した場合**は、それ以降の段階のキャッシュは、使えなくなります。
- 自分のパソコンでビルドする場合は、キャッシュが自動で効き、失敗した行の手前までは再利用されます。

## 3. Docker Hubの公開設定を確認する

Docker Hub → Repositories → `room3d-wilddet3d` → Settings で、**Public** になっているか確認します。
(無料プランの非公開リポジトリは、1つまでの記載がある資料が多いです。非公開にする場合は、RunPod側に認証情報の登録が必要です。)

> **公開リポジトリには、秘密情報や、利用申請が必要なモデルの重み・データを入れないでください。**
> このDockerfileは、これらを入れない設計になっています。

## 4. RunPodでテンプレートを作る

RunPodコンソール → Templates → New Template(画面の表記は、多少違うことがあります)。

| 項目 | 設定 |
|---|---|
| Template Name | 任意(例: `room3d-wilddet3d`) |
| Container Image | `ユーザー名/room3d-wilddet3d:1.0` |
| Container Disk | 40GB(余裕を持たせる) |
| Expose HTTP Ports | `8888` |
| Container Start Command | **空のまま**(イメージ内の `start.sh` が使われる) |
| Environment Variables | `JUPYTER_TOKEN` = 自分で決めた長い文字列(JupyterLabのログイン用) |

> Hugging Faceのトークンは、テンプレートには**保存しない**ことをおすすめします。ノートブックのSection 2で、実行時に入力します。

## 5. Podを立てる

1. Pods → Deploy → GPUを選ぶ(**Ampere / Ada / Hopper世代**。RTX 4090・A5000・A40・L4など。Blackwell世代は不可)。
2. テンプレートに、4.で作ったものを選ぶ。
3. **Persistent storage**(Network Volume)を作り、**30〜40GB**、マウント先は `/workspace`。
   (先に、選びたいGPUが空いているデータセンターを確認してから作るのがコツです。)
4. Deploy。
5. 起動したら、Connect → HTTP Service [Port 8888] を開き、`JUPYTER_TOKEN` に設定した文字列でログインする。
   (`JUPYTER_TOKEN` を設定しなかった場合は、Podのログにランダムなトークンが表示されます。)

## 6. ノートブックを使う

1. JupyterLabのファイルパネルで、`/workspace/` に `3dscenerestruction_walls_only_image.ipynb` をアップロードして開く。
2. `/workspace/inputs/` に、`room4.png` と `furniture_db_hssd_size.npz` をアップロードする。
3. 上のセルから順に実行する。Section 2で、Hugging Faceのトークンを貼り付ける。

初回は、モデルの重みのダウンロード(`/workspace/model_cache_export/` に保存)が発生します。2回目以降は、保存済みの重みを読み込むだけです。

## 7. 更新するとき

| 変えるもの | 作業 |
|---|---|
| ノートブックのコード | イメージの作り直しは**不要**。`/workspace` のノートブックを直すだけ |
| pipパッケージの追加・バージョン変更、WildDet3D・vis4d_cuda_opsの更新 | `Dockerfile` を直して、`docker build -t ユーザー名/room3d-wilddet3d:1.1 .` → `docker push` → RunPodのテンプレートの `Container Image` を `:1.1` に更新 |

Dockerは、変わっていない層を再利用するため、`Dockerfile` の後ろのほうだけを変えた場合は、ビルドもPushも短く済みます。

## 8. うまくいかないとき

| 症状 | 原因の候補と対処 |
|---|---|
| ビルドが「no space left on device」で失敗する | ディスクの空きが足りない。Docker Desktopの割り当て容量を増やす、または `docker system prune -a` で不要なイメージを消す |
| Podが起動せず、イメージ取得のエラーが出る | Docker Hubの取得回数の制限、または公開設定の間違い。少し待つか、別のGPU・リージョンで作り直す |
| JupyterLabにログインできない | `JUPYTER_TOKEN` の設定を確認する。未設定なら、Podのログに表示されたトークンを使う |
| ノートブックの最初のセルで「専用イメージ用です」と出る | 専用イメージ以外のPodで開いている。`3dscenerestruction_walls_only_runpod.ipynb` を使う |
| 実行時に「no kernel image is available」と出る | このGPUの世代が、`TORCH_CUDA_ARCH_LIST` に含まれていない。`Dockerfile` の値に追加して、作り直す |
| 「このGPUはBlackwell世代です」と出る | torch 2.5.1が非対応。別世代のGPUでPodを作り直す |
