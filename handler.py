"""
RunPod Serverless のエントリーポイント。
起動時(コールドスタート)に pipeline.load_models() を1回だけ呼び、
以降のリクエストは pipeline.run_pipeline() を呼ぶだけにする(ウォームな
ワーカーでは、モデルの再読み込みをしない)。

重要: pipeline.py の Section 13(CoACD)は、multiprocessing.get_context("spawn") で
子プロセスを起動する。spawn方式では、子プロセスは「__main__」を安全に再importするため、
このファイル(python handler.py で実行される、__main__そのもの)を、子プロセスも
もう一度読み込み直す。そのため、モデルの読み込みやワーカーの起動といった「本当に一度
きりでよい処理」は、必ず `if __name__ == "__main__":` の中に置く必要がある。
このガードが無いと、CoACDの並列数(例: 12)ぶん、コールドストート(モデルの再読み込み)が
繰り返し発生してしまう(実際に起きた不具合)。
参考: https://docs.python.org/ja/3/library/multiprocessing.html#the-spawn-and-forkserver-start-methods

未検証: ローカルでの動作確認(このファイル末尾の実行例)を、必ず先に行ってください。
"""
import base64
import traceback

import runpod

import pipeline


def handler(job):
    job_input = job.get("input", {}) or {}

    image_b64 = job_input.get("image_base64")
    if not image_b64:
        return {"error": "input.image_base64 が指定されていません(部屋の写真をbase64で渡してください)。"}

    try:
        image_bytes = base64.b64decode(image_b64)
    except Exception as e:
        return {"error": f"input.image_base64 のデコードに失敗しました: {e!r}"}

    options = {
        "coacd_fast_mode": job_input.get("coacd_fast_mode", True),
    }

    def progress(payload):
        # payload は {"percent": int, "label": str}
        runpod.serverless.progress_update(job, payload)

    try:
        result = pipeline.run_pipeline(image_bytes, options=options, progress=progress)
    except Exception as e:
        # スタックトレースをログに残しつつ、呼び出し側にもエラー内容を返す
        traceback.print_exc()
        return {"error": f"{type(e).__name__}: {e}"}

    return result


if __name__ == "__main__":
    # このガードのおかげで、CoACD(spawn方式)の子プロセスが本ファイルを再読み込みしても、
    # ここから下は実行されない(子プロセスでの__name__は "__main__" にならないため)。
    print("=== コールドスタート: モデルを読み込みます ===")
    pipeline.load_models()
    print("=== モデルの読み込みが完了しました。リクエスト受付を開始します ===")

    runpod.serverless.start({"handler": handler})
