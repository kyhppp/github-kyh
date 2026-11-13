"""AI 모델 다운로드 및 ONNX 세션 관리."""
from __future__ import annotations

import os
import sys
import threading
from pathlib import Path

import onnxruntime as ort
import requests

MODEL_DIR = Path(__file__).resolve().parent.parent / "models"

_FF = "https://github.com/facefusion/facefusion-assets/releases/download/models-3.0.0/"
MODELS = {
    "real_esrgan_x4": _FF + "real_esrgan_x4.onnx",
    "real_esrgan_x2": _FF + "real_esrgan_x2.onnx",
    # 사진 질감(옷감·머리카락·수염)을 덜 뭉개는 4배 업스케일러
    "nomos8k_sc_x4": _FF + "nomos8k_sc_x4.onnx",
    "gfpgan_1.4": _FF + "gfpgan_1.4.onnx",
    "codeformer": _FF + "codeformer.onnx",
    "ddcolor": _FF + "ddcolor.onnx",
    "ddcolor_artistic": _FF + "ddcolor_artistic.onnx",
    "yunet": "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx",
    "lama": "https://huggingface.co/Carve/LaMa-ONNX/resolve/main/lama_fp32.onnx",
}

_lock = threading.Lock()
_sessions: dict[tuple[str, bool], ort.InferenceSession] = {}


def model_path(name: str, progress=None) -> Path:
    """모델 파일 경로를 돌려주고, 없으면 내려받는다."""
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    path = MODEL_DIR / f"{name}.onnx"
    if path.exists():
        return path
    url = MODELS[name]
    tmp = path.with_suffix(".part")
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        done = 0
        with open(tmp, "wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 20):
                f.write(chunk)
                done += len(chunk)
                if progress and total:
                    progress(done / total, desc=f"모델 다운로드 중: {name} ({done >> 20}/{total >> 20} MB)")
                elif total and progress is None:
                    sys.stdout.write(f"\r  {name}: {done >> 20}/{total >> 20} MB")
                    sys.stdout.flush()
    if progress is None:
        print()
    os.replace(tmp, path)
    return path


def use_gpu_default() -> bool:
    return "DmlExecutionProvider" in ort.get_available_providers()


# DirectML에서 지원되지 않는 연산(FFT 등)을 쓰는 모델은 CPU로만 실행
CPU_ONLY = {"lama"}


def session(name: str, gpu: bool = True, progress=None) -> ort.InferenceSession:
    gpu = gpu and use_gpu_default() and name not in CPU_ONLY
    key = (name, gpu)
    with _lock:
        if key not in _sessions:
            path = model_path(name, progress)
            opts = ort.SessionOptions()
            opts.log_severity_level = 3
            providers = ["DmlExecutionProvider", "CPUExecutionProvider"] if gpu else ["CPUExecutionProvider"]
            if gpu:
                # DirectML은 메모리 패턴/병렬 실행을 지원하지 않음
                opts.enable_mem_pattern = False
                opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
            _sessions[key] = ort.InferenceSession(str(path), sess_options=opts, providers=providers)
        return _sessions[key]


def run(name: str, feeds: dict, gpu: bool = True, progress=None):
    """모델 실행. GPU(DirectML)에서 실패하면 CPU로 다시 시도한다."""
    sess = session(name, gpu, progress)
    try:
        return sess.run(None, feeds)
    except Exception:
        if not (gpu and use_gpu_default()):
            raise
        return session(name, False).run(None, feeds)


def input_info(name: str, gpu: bool = True, progress=None):
    return session(name, gpu, progress).get_inputs()


if __name__ == "__main__":
    # python -m restorer.models  → 모든 모델 미리 받기
    for n in (sys.argv[1:] or MODELS):
        print(n, "->", model_path(n))
