"""사진 복원 파이프라인.

모든 이미지는 BGR uint8 (OpenCV 형식)으로 다룬다.
처리 순서: 긁힘·얼룩 제거 → 색 바램 보정 → 업스케일 → 얼굴 복원 → 컬러화
"""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from .models import input_info, model_path, run


@dataclass
class Options:
    # 긁힘·얼룩
    scratch_auto: bool = False
    scratch_sensitivity: float = 0.5   # 0~1, 클수록 더 많이 잡음
    scratch_method: str = "lama"       # "lama" | "opencv"
    user_mask: np.ndarray | None = None  # 브러시로 칠한 영역 (H,W) uint8
    # 색 바램 보정
    auto_tone: bool = False
    # 업스케일
    upscale: int = 2                   # 1, 2, 4
    upscale_model: str = "nomos8k_sc_x4"  # "nomos8k_sc_x4"(디테일) | "real_esrgan"(매끈)
    # 얼굴
    face_restore: bool = True
    face_model: str = "gfpgan_1.4"     # "gfpgan_1.4" | "codeformer"
    face_strength: float = 0.8         # 원본과 섞는 비율 (1 = 복원본만)
    codeformer_fidelity: float = 0.7   # 0=화질 우선, 1=원본 닮음 우선
    # 컬러화
    colorize: bool = False
    color_model: str = "ddcolor"       # "ddcolor" | "ddcolor_artistic"
    # 장치
    gpu: bool = True
    debug: dict = field(default_factory=dict)


def _noop(*_a, **_k):
    return None


# ---------------------------------------------------------------- 긁힘·얼룩

def _extend_line(support: np.ndarray, p1, p2, gap: int):
    """선분 양 끝을 같은 방향으로, support 픽셀이 gap 이내로 이어지는 동안 늘린다."""
    h, w = support.shape
    d = np.array(p2, np.float32) - np.array(p1, np.float32)
    n = float(np.hypot(*d))
    if n < 1:
        return tuple(p1), tuple(p2)
    d /= n
    normal = np.array([-d[1], d[0]])

    def walk(start, direction):
        pos = np.array(start, np.float32)
        last, miss = pos.copy(), 0
        while miss <= gap:
            pos = pos + direction
            found = False
            for off in (0, 1, -1):  # 선 옆 1px까지 허용
                x, y = (pos + off * normal).round().astype(int)
                if not (0 <= x < w and 0 <= y < h):
                    return tuple(last.round().astype(int))
                if support[y, x]:
                    found = True
                    break
            if found:
                last, miss = pos.copy(), 0
            else:
                miss += 1
        return tuple(int(v) for v in last.round())

    return walk(p1, -d), walk(p2, d)


def detect_scratches(img: np.ndarray, sensitivity: float = 0.5, protect_faces: bool = True) -> np.ndarray:
    """얇은 밝은/어두운 선(긁힘)과 작은 점(먼지)을 찾아 마스크(0/255)로 돌려준다."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    scale = max(1.0, max(h, w) / 800.0)       # 큰 사진일수록 긁힘도 굵다
    k = int(round(5 * scale)) | 1             # 긁힘 굵기보다 넓은 미디언 창

    # 1) 미디언 필터는 얇은 선·점을 지우고 경계는 보존 → 원본과의 차이 = 긁힘 후보
    med = cv2.medianBlur(gray, min(k, 255))
    diff = gray.astype(np.float32) - med.astype(np.float32)

    # 2) 질감이 많은 곳(머리카락, 벽돌, 옷감)은 더 큰 차이가 있어야 긁힘으로 본다
    mf = med.astype(np.float32)
    activity = cv2.GaussianBlur(np.abs(mf - cv2.GaussianBlur(mf, (0, 0), 1.5 * scale)), (0, 0), 3 * scale)
    t = 1.0 - sensitivity                     # 0(관대) ~ 1(엄격)
    floor = 18.0 + 30.0 * t                   # 최소 밝기 차이
    ratio = 3.0 + 5.0 * t                     # 주변 질감 대비 배수
    strong = np.abs(diff) > ratio * activity + 4
    mask = np.zeros((h, w), np.uint8)       # 점·짧은 선 (얼굴 안에서는 무시)
    line_mask = np.zeros((h, w), np.uint8)  # 길게 확인된 직선 긁힘 (얼굴 위도 지움)
    dust_max = int(12 * scale * scale)
    hit_need = 0.85 - 0.3 * sensitivity     # 직선을 따라 후보가 이어진 비율

    # 얼굴 영역: 눈·입·주름은 먼지·짧은 선과 비슷해 보이므로 따로 다룬다
    face_area = np.zeros((h, w), np.uint8)
    face_eye = np.zeros((h, w), np.float32)  # 그 위치 얼굴의 눈 사이 거리
    if protect_faces:
        for lm in detect_faces(img, 0.5):
            c = lm.mean(0)
            eye = float(np.linalg.norm(lm[1] - lm[0]))
            ell = np.zeros((h, w), np.uint8)
            cv2.ellipse(ell, (int(c[0]), int(c[1])), (int(eye * 1.3), int(eye * 1.7)), 0, 0, 360, 1, -1)
            face_area |= ell
            face_eye = np.maximum(face_eye, ell * eye)
    for polarity in (1, -1):  # 밝은 긁힘 / 어두운 긁힘을 따로 본다
        cand = ((polarity * diff > floor) & strong).astype(np.uint8)
        loose = ((polarity * diff > floor * 0.6) & (np.abs(diff) > 0.6 * ratio * activity)).astype(np.uint8)

        # 3) 모양 검사: 가늘고 긴 선(긁힘)이나 작은 점(먼지)만 남김
        n, labels, stats, _ = cv2.connectedComponentsWithStats(cand, connectivity=8)
        keep = np.zeros(n, dtype=bool)
        for i in range(1, n):
            x, y, bw, bh, area = stats[i]
            if area <= dust_max:
                peak = np.abs(diff[y:y + bh, x:x + bw][labels[y:y + bh, x:x + bw] == i]).max()
                keep[i] = peak > floor * 1.6  # 먼지는 아주 뚜렷해야 함
                continue
            pts = np.column_stack(np.nonzero(labels[y:y + bh, x:x + bw] == i)).astype(np.float32)
            (_, _), (rw, rh), _ = cv2.minAreaRect(pts)
            length = max(rw, rh, 1.0)
            keep[i] = length >= 8 * scale and area / length <= 2.5 * scale
        mask |= keep[labels].astype(np.uint8)

        # 4) 길게 이어진 직선 긁힘은 허프 변환으로 따로 찾아 끊긴 부분까지 잇는다
        min_len = max(20, int(0.07 * max(h, w)))
        lines = cv2.HoughLinesP(loose * 255, 1, np.pi / 360, threshold=max(12, int(min_len * 0.5)),
                                minLineLength=min_len, maxLineGap=int(8 * scale))
        if lines is not None:
            loose_d = cv2.dilate(loose, np.ones((3, 3), np.uint8))
            for x1, y1, x2, y2 in np.asarray(lines).reshape(-1, 4).astype(int):
                # 선을 따라 후보 픽셀이 충분히 있을 때만 긁힘으로 인정 (1px 선으로 측정)
                thin = np.zeros_like(mask)
                cv2.line(thin, (x1, y1), (x2, y2), 1, 1)
                hit = (thin & loose_d).sum() / max(thin.sum(), 1)
                need = hit_need
                on_line = thin > 0
                if face_area[on_line].all():
                    # 얼굴 안에만 있는 선은 눈썹·주름보다 확실히 길고 끊김이 없어야 긁힘
                    if np.hypot(x2 - x1, y2 - y1) < 1.5 * float(face_eye[on_line].max()):
                        continue
                    need = max(need, 0.8)
                if hit > need:
                    # 같은 방향으로 후보가 이어지는 동안 선을 연장 (얼굴 위로 이어진 부분까지)
                    p1, p2 = _extend_line(loose_d, (x1, y1), (x2, y2), gap=int(6 * scale))
                    one = np.zeros_like(mask)
                    cv2.line(one, p1, p2, 1, max(1, int(round(scale))) + 2)
                    line_mask |= one & loose_d

    # 5) 얼굴 안에서는 점·짧은 선은 지우지 않고, 긴 직선 긁힘만 지운다
    mask[face_area > 0] = 0

    r = max(1, int(round(scale)))
    return cv2.dilate((mask | line_mask) * 255, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1)))


def _lama_tile(img: np.ndarray, mask: np.ndarray, gpu: bool) -> np.ndarray:
    h, w = img.shape[:2]
    rgb = cv2.resize(img, (512, 512), interpolation=cv2.INTER_AREA)[:, :, ::-1]
    m = cv2.resize(mask, (512, 512), interpolation=cv2.INTER_NEAREST)
    x = (rgb.astype(np.float32) / 255.0).transpose(2, 0, 1)[None]
    mm = (m > 127).astype(np.float32)[None, None]
    out = run("lama", {"image": x, "mask": mm}, gpu)[0][0]
    if out.max() <= 1.5:
        out = out * 255.0
    out = np.clip(out.transpose(1, 2, 0), 0, 255).astype(np.uint8)[:, :, ::-1]
    return cv2.resize(out, (w, h), interpolation=cv2.INTER_CUBIC)


def inpaint(img: np.ndarray, mask: np.ndarray, method: str = "lama", gpu: bool = True, progress=_noop) -> np.ndarray:
    if not mask.any():
        return img
    if method == "opencv":
        return cv2.inpaint(img, mask, 3, cv2.INPAINT_TELEA)

    out = img.copy()
    h, w = img.shape[:2]
    # 큰 사진은 LaMa 해상도(512)에 맞게 작업 배율을 정한다. 1024px 이하 영역을 512 타일로 처리.
    tile = 512 if max(h, w) <= 2048 else 768
    step = tile - 128
    ys = list(range(0, max(h - tile, 0) + 1, step))
    xs = list(range(0, max(w - tile, 0) + 1, step))
    if ys[-1] + tile < h:
        ys.append(max(h - tile, 0))
    if xs[-1] + tile < w:
        xs.append(max(w - tile, 0))
    jobs = [(y, x) for y in ys for x in xs if mask[y:y + tile, x:x + tile].any()]
    for i, (y, x) in enumerate(jobs):
        progress(i / max(len(jobs), 1), desc=f"긁힘·얼룩 메우는 중 ({i + 1}/{len(jobs)})")
        crop = out[y:y + tile, x:x + tile]
        mcrop = mask[y:y + tile, x:x + tile]
        filled = _lama_tile(crop, mcrop, gpu)
        soft = cv2.GaussianBlur((mcrop > 127).astype(np.float32), (5, 5), 0)
        soft = np.maximum(soft, (mcrop > 127).astype(np.float32))[:, :, None]
        out[y:y + tile, x:x + tile] = (filled * soft + crop * (1 - soft)).astype(np.uint8)
    return out


# ---------------------------------------------------------------- 색 바램 보정

def auto_tone(img: np.ndarray) -> np.ndarray:
    """바랜 사진의 대비·색 균형을 살짝 되살린다."""
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    lo, hi = np.percentile(l, (0.5, 99.5))
    if hi - lo > 10:
        l = np.clip((l.astype(np.float32) - lo) * 255.0 / (hi - lo), 0, 255).astype(np.uint8)
    l = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8)).apply(l)
    # 누렇게 바랜 색조(세피아)를 중립 쪽으로 절반만 당김
    a = np.clip(a.astype(np.float32) - (a.mean() - 128) * 0.5, 0, 255).astype(np.uint8)
    b = np.clip(b.astype(np.float32) - (b.mean() - 128) * 0.5, 0, 255).astype(np.uint8)
    return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)


# ---------------------------------------------------------------- 업스케일

def upscale(img: np.ndarray, factor: int, model: str = "nomos8k_sc_x4", gpu: bool = True,
            progress=_noop) -> np.ndarray:
    """factor배 확대. 4배 전용 모델로 2배를 만들 땐 4배로 키운 뒤 줄인다 (줄이면서 더 또렷해짐)."""
    if factor <= 1:
        return img
    if model == "real_esrgan":
        return _sr(img, f"real_esrgan_x{factor}", factor, gpu, progress)
    out = _sr(img, model, 4, gpu, progress)
    if factor != 4:
        out = cv2.resize(out, (img.shape[1] * factor, img.shape[0] * factor), interpolation=cv2.INTER_AREA)
    return out


def _sr(img: np.ndarray, mname: str, factor: int, gpu: bool = True, progress=_noop) -> np.ndarray:
    inp = input_info(mname, gpu, progress)[0]
    tile, pad = 256, 16
    h, w = img.shape[:2]
    padded = cv2.copyMakeBorder(img, pad, pad + tile, pad, pad + tile, cv2.BORDER_REFLECT_101)
    out = np.zeros((h * factor, w * factor, 3), dtype=np.uint8)
    tiles = [(y, x) for y in range(0, h, tile) for x in range(0, w, tile)]
    for i, (y, x) in enumerate(tiles):
        progress(i / len(tiles), desc=f"해상도 높이는 중 ({i + 1}/{len(tiles)})")
        crop = padded[y:y + tile + 2 * pad, x:x + tile + 2 * pad]
        t = (crop[:, :, ::-1].astype(np.float32) / 255.0).transpose(2, 0, 1)[None]
        r = run(mname, {inp.name: t}, gpu)[0][0]
        r = (np.clip(r.transpose(1, 2, 0), 0, 1) * 255.0).round().astype(np.uint8)[:, :, ::-1]
        r = r[pad * factor:(pad + tile) * factor, pad * factor:(pad + tile) * factor]
        th, tw = min(tile, h - y), min(tile, w - x)
        out[y * factor:(y + th) * factor, x * factor:(x + tw) * factor] = r[:th * factor, :tw * factor]
    return out


# ---------------------------------------------------------------- 얼굴 복원

FFHQ_512 = np.array([
    [0.37691676, 0.46864664],
    [0.62285697, 0.46912813],
    [0.50123859, 0.61331904],
    [0.39308822, 0.72541100],
    [0.61150205, 0.72490465],
], dtype=np.float32) * 512


def detect_faces(img: np.ndarray, score: float = 0.6) -> list[np.ndarray]:
    """얼굴마다 5개 특징점 (눈2, 코, 입꼬리2) 배열을 돌려준다."""
    h, w = img.shape[:2]
    # YuNet은 너무 큰 입력에서 느리므로 축소해서 찾는다
    s = min(1.0, 1280.0 / max(h, w))
    small = cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA) if s < 1 else img
    # OpenCV는 한글 경로를 못 읽으므로 모델을 메모리 버퍼로 넘긴다
    buf = np.frombuffer(model_path("yunet").read_bytes(), np.uint8)
    det = cv2.FaceDetectorYN.create("onnx", buf, np.empty(0, np.uint8), (small.shape[1], small.shape[0]), score, 0.3, 5000)
    _, faces = det.detect(small)
    if faces is None:
        return []
    result = []
    for f in faces:
        if f[2] * f[3] / (s * s) < 24 * 24:  # 너무 작은 얼굴은 건너뜀
            continue
        result.append(f[4:14].reshape(5, 2) / s)
    return result


def _face_mask() -> np.ndarray:
    m = np.zeros((512, 512), np.float32)
    pad = 26
    m[pad:-pad, pad:-pad] = 1.0
    return cv2.GaussianBlur(m, (0, 0), 14)


def restore_faces(img: np.ndarray, opt: Options, progress=_noop) -> np.ndarray:
    faces = detect_faces(img)
    opt.debug["faces"] = len(faces)
    if not faces:
        return img
    names = [i.name for i in input_info(opt.face_model, opt.gpu, progress)]
    out = img.astype(np.float32)
    h, w = img.shape[:2]
    base_mask = _face_mask()
    for i, lm in enumerate(faces):
        progress(i / len(faces), desc=f"얼굴 복원 중 ({i + 1}/{len(faces)})")
        M, _ = cv2.estimateAffinePartial2D(lm.astype(np.float32), FFHQ_512, method=cv2.LMEDS)
        if M is None:
            continue
        crop = cv2.warpAffine(img, M, (512, 512), borderMode=cv2.BORDER_REFLECT_101, flags=cv2.INTER_AREA)
        x = ((crop[:, :, ::-1].astype(np.float32) / 255.0 - 0.5) / 0.5).transpose(2, 0, 1)[None]
        feeds = {}
        for n in names:
            if n == "weight":
                feeds[n] = np.array(opt.codeformer_fidelity, dtype=np.float64)
            else:
                feeds[n] = x
        r = run(opt.face_model, feeds, opt.gpu)[0][0]
        r = ((np.clip(r, -1, 1) + 1) / 2).transpose(1, 2, 0)
        r = (r * 255.0).round().astype(np.uint8)[:, :, ::-1]
        fmask = base_mask
        # 사진 속 얼굴이 512px보다 크면 복원된 얼굴을 그냥 늘려 붙이면 흐려진다.
        # → 복원된 얼굴도 업스케일러로 키운 다음 실제 크기에 맞춰 붙인다.
        k = 1.0 / np.sqrt(abs(np.linalg.det(M[:, :2])))  # 512px 얼굴 1px = 사진 k px
        if k > 1.25:
            progress(i / len(faces), desc=f"큰 얼굴 고해상도로 마무리 ({i + 1}/{len(faces)})")
            size = int(round(512 * min(k, 4.0)))
            r4 = upscale(r, 4, opt.upscale_model, opt.gpu)
            r = cv2.resize(r4, (size, size), interpolation=cv2.INTER_AREA)
            fmask = cv2.resize(base_mask, (size, size), interpolation=cv2.INTER_LINEAR)
            M = M * (size / 512.0)
        IM = cv2.invertAffineTransform(M)
        back = cv2.warpAffine(r, IM, (w, h), borderMode=cv2.BORDER_REPLICATE, flags=cv2.INTER_LINEAR)
        mask = cv2.warpAffine(fmask, IM, (w, h))[:, :, None] * opt.face_strength
        out = back.astype(np.float32) * mask + out * (1 - mask)
    return np.clip(out, 0, 255).round().astype(np.uint8)


# ---------------------------------------------------------------- 컬러화

def colorize(img: np.ndarray, model: str = "ddcolor", gpu: bool = True, progress=_noop) -> np.ndarray:
    inp = input_info(model, gpu, progress)[0]
    size = inp.shape[2] if isinstance(inp.shape[2], int) else 512
    progress(0.5, desc="컬러 입히는 중")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB).astype(np.float32) / 255.0
    l = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)[:, :, :1]
    gray_rgb = cv2.cvtColor(np.concatenate([l, np.zeros_like(l), np.zeros_like(l)], -1), cv2.COLOR_LAB2RGB)
    x = cv2.resize(gray_rgb, (size, size), interpolation=cv2.INTER_AREA).transpose(2, 0, 1)[None].astype(np.float32)
    ab = run(model, {inp.name: x}, gpu)[0][0].transpose(1, 2, 0)
    ab = cv2.resize(ab, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_CUBIC)
    # 원본 밝기(L)는 그대로 두고 색(ab)만 입힌다
    l_full = cv2.cvtColor(img.astype(np.float32) / 255.0, cv2.COLOR_BGR2LAB)[:, :, :1]
    lab = np.concatenate([l_full, ab], -1)
    out = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    return (np.clip(out, 0, 1) * 255.0).round().astype(np.uint8)


# ---------------------------------------------------------------- 전체

def restore(img: np.ndarray, opt: Options, progress=_noop) -> np.ndarray:
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

    mask = np.zeros(img.shape[:2], np.uint8)
    if opt.scratch_auto:
        progress(0.0, desc="긁힘·얼룩 찾는 중")
        mask |= detect_scratches(img, opt.scratch_sensitivity)
    if opt.user_mask is not None and opt.user_mask.shape == mask.shape:
        mask |= (opt.user_mask > 0).astype(np.uint8) * 255
    opt.debug["mask"] = mask
    if mask.any():
        img = inpaint(img, mask, opt.scratch_method, opt.gpu, progress)

    if opt.auto_tone:
        img = auto_tone(img)
    if opt.upscale > 1:
        img = upscale(img, opt.upscale, opt.upscale_model, opt.gpu, progress)
    if opt.face_restore:
        img = restore_faces(img, opt, progress)
    if opt.colorize:
        img = colorize(img, opt.color_model, opt.gpu, progress)
    return img
