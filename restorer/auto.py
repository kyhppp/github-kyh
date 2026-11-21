"""'✨ 자동' 버튼: 사진을 분석해서 각 설정의 추천값을 정한다.

추천값은 출발점일 뿐이라, 사용자가 보고 더 조절하는 것을 전제로 무난한 쪽으로 고른다.
모든 함수는 BGR uint8 사진을 받아 (값 dict, 사람이 읽을 설명) 을 돌려준다.
"""
from __future__ import annotations

import cv2
import numpy as np

from . import pipeline as P
from . import trim as T

ANALYZE_SIDE = 1600  # 분석은 이 크기로 줄여서 (큰 사진도 1초 안쪽)


def _small(img: np.ndarray, side: int = ANALYZE_SIDE) -> np.ndarray:
    h, w = img.shape[:2]
    s = side / max(h, w)
    if s >= 1:
        return img
    return cv2.resize(img, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)


def _luma(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0


def _saturation(img: np.ndarray) -> float:
    """평균 채도 0~1 (흑백·세피아 사진은 아주 낮다)"""
    return float(cv2.cvtColor(img, cv2.COLOR_BGR2HSV)[:, :, 1].mean() / 255.0)


def is_grayscale(img: np.ndarray) -> bool:
    """흑백(또는 단색으로 바랜 세피아) 사진인지. 채널 차이가 거의 없으면 흑백."""
    x = _small(img, 600).astype(np.int16)
    spread = (x.max(axis=2) - x.min(axis=2)).astype(np.float32)
    return float(np.percentile(spread, 95)) < 28 and _saturation(_small(img, 600)) < 0.12


# ------------------------------------------------------------------ 복원 옵션

def suggest_scratch(img: np.ndarray):
    small = _small(img, 1000)  # 판단용이라 작게 (큰 사진도 2초 안쪽)
    pct = float((P.detect_scratches(small, 0.4) > 0).mean() * 100)
    raw = float((P.detect_scratches(small, 0.4, busy_density=None) > 0).mean() * 100)  # 무늬 제외 전
    if pct < 0.02:
        return dict(on=False, sensitivity=0.4, method="lama"), \
            f"긁힘·얼룩이 거의 안 보여서 자동 제거를 껐어요 (감지 {pct:.2f}%)."
    if raw > 5.0:  # 무늬 제외 전에 이만큼 잡히면 옷감·나뭇잎 같은 무늬가 많은 사진
        return dict(on=False, sensitivity=0.4, method="lama"), \
            ("무늬가 많은 사진이라 자동 감지가 엉뚱한 곳까지 잡을 수 있어요. 자동 제거는 껐으니, "
             "긁힌 곳이 있으면 브러시로 칠해 주세요.")
    if pct > 2.0:  # 꽤 많이 잡히면 일부는 실제 무늬일 가능성 → 감도를 낮춘다
        return dict(on=True, sensitivity=0.25, method="lama"), \
            f"감지된 부분이 많아({pct:.1f}%) 실제 무늬까지 지우지 않게 감도를 낮췄어요. '자동 감지 미리보기'로 확인해 보세요."
    return dict(on=True, sensitivity=0.4, method="lama"), \
        f"긁힘·얼룩이 {pct:.2f}% 보여서 자동 제거를 켰어요. '자동 감지 미리보기'로 확인해 보세요."


def suggest_tone(img: np.ndarray):
    small = _small(img, 800)
    L = _luma(small)
    lo, hi = np.percentile(L, [1, 99])
    rng = float(hi - lo)
    # 누런 색조: 파랑 채널이 빨강보다 많이 약함 (흑백 사진은 색조 판단에서 제외)
    b, g, r = (small.reshape(-1, 3).astype(np.float32).mean(axis=0) / 255.0)
    yellow = (r - b) > 0.10 and not is_grayscale(img)
    if rng < 0.70 or yellow:
        why = []
        if rng < 0.70:
            why.append(f"대비가 약하고(밝기 폭 {rng * 100:.0f}%)")
        if yellow:
            why.append("누런 색조가 있어서")
        return dict(on=True), " ".join(why) + " 색 바램 보정을 켰어요."
    return dict(on=False), "대비와 색이 괜찮아서 색 바램 보정은 껐어요."


def suggest_upscale(img: np.ndarray):
    h, w = img.shape[:2]
    f = P.upscale_factor(h, w, 0)
    # 잡티(고주파 노이즈)가 많으면 '매끈하게'가 더 깔끔하다
    L = _luma(_small(img, 1200))
    noise = float(np.median(np.abs(L - cv2.medianBlur((L * 255).astype(np.uint8), 3).astype(np.float32) / 255.0)))
    model = "real_esrgan" if noise > 0.025 else "nomos8k_sc_x4"
    msg = (f"사진이 {w}×{h}라 " + (f"{f:.1f}배로 키워요." if f > 1 else "이미 충분히 커서 키우지 않아요 (빠름)."))
    msg += " 잡티가 많아 '매끈하게' 방식을 골랐어요." if model == "real_esrgan" else ""
    return dict(choice=0, model=model), msg


def suggest_face(img: np.ndarray):
    faces = P.detect_faces(img)
    if not faces:
        return dict(on=False, model="gfpgan_1.4", strength=0.85, fidelity=0.7), \
            "얼굴을 찾지 못해서 얼굴 복원을 껐어요."
    eyes = [float(np.linalg.norm(lm[1] - lm[0])) for lm in faces]
    small_face = float(np.median(eyes)) < 40  # 눈 사이 40px 미만 = 작고 뭉개진 얼굴이 많음
    # 얼굴 선명도: 얼굴 주변 라플라시안 분산이 낮으면 흐린 얼굴
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sharp = []
    for lm, e in zip(faces, eyes):
        c = lm.mean(0).astype(int)
        r = max(8, int(e * 1.2))
        crop = gray[max(0, c[1] - r):c[1] + r, max(0, c[0] - r):c[0] + r]
        if crop.size:
            crop = cv2.resize(crop, (128, 128), interpolation=cv2.INTER_AREA)
            sharp.append(cv2.Laplacian(crop, cv2.CV_64F).var())
    blurry = bool(sharp) and float(np.median(sharp)) < 60
    n = len(faces)
    if small_face or blurry:
        return dict(on=True, model="codeformer", strength=0.9, fidelity=0.6), \
            f"얼굴 {n}개를 찾았어요. 작거나 흐릿해서 CodeFormer(손상 심한 얼굴용)로 골랐어요."
    return dict(on=True, model="gfpgan_1.4", strength=0.7, fidelity=0.7), \
        f"얼굴 {n}개를 찾았어요. 비교적 또렷해서 자연스러운 GFPGAN에 강도를 조금 낮췄어요."


def suggest_colorize(img: np.ndarray):
    if is_grayscale(img):
        return dict(on=True, model="ddcolor"), "흑백 사진이라 컬러화를 켰어요."
    return dict(on=False, model="ddcolor"), "이미 컬러 사진이라 컬러화는 껐어요 (켜면 원래 색이 바뀌어요)."


# ------------------------------------------------------------------ 마무리 (색감·트리밍)

def suggest_light(img: np.ndarray):
    L = _luma(_small(img, 800))
    mid = float(np.median(L))  # 흰 배경·검은 옷 하나에 끌려가지 않게 평균 대신 중앙값
    lo, hi = np.percentile(L, [1, 99])
    # 확실히 어둡거나 밝을 때만 노출을 건드린다 (적당한 사진은 그대로)
    if mid < 0.33:
        exposure = float(np.clip(np.log2(0.42 / max(mid, 0.02)) / 1.5 * 100, 0, 50))
    elif mid > 0.68:
        exposure = float(np.clip(np.log2(0.60 / mid) / 1.5 * 100, -35, 0))
    else:
        exposure = 0.0
    rng = float(hi - lo)
    contrast = float(np.clip((0.85 - rng) * 120, 0, 35)) if rng < 0.85 else 0.0
    clipped_hi = float((L > 0.97).mean())
    clipped_lo = float((L < 0.06).mean())
    highlights = -float(np.clip(clipped_hi * 600, 0, 50))
    shadows = float(np.clip(clipped_lo * 400, 0, 45))
    # 아주 작은 조정은 안 하느니만 못해서 버린다
    vals = {k: (round(v) if abs(v) >= 5 else 0) for k, v in
            dict(exposure=exposure, contrast=contrast, highlights=highlights, shadows=shadows).items()}
    parts = []
    if abs(vals["exposure"]) >= 3:
        parts.append("밝게" if vals["exposure"] > 0 else "어둡게")
    if vals["contrast"]:
        parts.append("대비 올림")
    if vals["highlights"]:
        parts.append("날아간 밝은 부분 살림")
    if vals["shadows"]:
        parts.append("그늘 밝힘")
    return vals, ("빛: " + ", ".join(parts)) if parts else "빛: 밝기와 대비가 이미 적당해요."


def suggest_color(img: np.ndarray):
    small = _small(img, 800)
    if is_grayscale(small):
        return dict(auto_wb=False, temperature=0, tint=0, saturation=0, vibrance=0), \
            "색: 흑백 사진이라 색은 건드리지 않았어요."
    x = small.astype(np.float32) / 255.0
    L = x @ np.array([0.114, 0.587, 0.299], np.float32)
    sel = (L > np.percentile(L, 20)) & (L < np.percentile(L, 99))
    mean = x[sel].mean(axis=0) if sel.sum() > 100 else x.reshape(-1, 3).mean(axis=0)
    cast = float(mean.max() - mean.min()) / max(float(mean.mean()), 1e-3)
    auto_wb = cast > 0.12
    sat = _saturation(small)
    vibrance = float(np.clip((0.32 - sat) * 120, 0, 35)) if sat < 0.32 else 0.0
    vals = dict(auto_wb=auto_wb, temperature=0, tint=0, saturation=0, vibrance=round(vibrance))
    parts = (["색이 한쪽으로 떠 있어 자동 색 균형 켬"] if auto_wb else []) + (["옅은 색 살림"] if vibrance else [])
    return vals, ("색: " + ", ".join(parts)) if parts else "색: 색 균형과 채도가 이미 적당해요."


def suggest_effects(img: np.ndarray):
    """취향 효과(세피아·바랜 느낌·비네팅)는 끄고, 흐릿할 때만 선명도를 조금 올린다."""
    L = (_luma(_small(img, 1200)) * 255).astype(np.uint8)
    blur = float(cv2.Laplacian(L, cv2.CV_64F).var())
    sharpness = 30 if blur < 80 else 15 if blur < 250 else 0
    msg = "효과: 세피아·바랜 느낌·비네팅은 끄고, " + (f"조금 흐릿해서 선명도 {sharpness}" if sharpness else "선명도는 그대로")
    return dict(sepia=0, fade=0, vignette=0, sharpness=sharpness), msg


def suggest_trim(img: np.ndarray):
    """테두리를 찾아 여백 슬라이더에 직접 넣는다 (보면서 더 조절할 수 있게 '자동 테두리'는 끔)."""
    t, b, l, r = T.detect_border(img)
    vals = dict(auto_border=False, top=round(t * 2) / 2, bottom=round(b * 2) / 2,
                left=round(l * 2) / 2, right=round(r * 2) / 2, ratio=T.FREE)
    if any((t, b, l, r)):
        return vals, f"트리밍: 테두리를 찾아 위 {vals['top']}% · 아래 {vals['bottom']}% · 왼쪽 {vals['left']}% · 오른쪽 {vals['right']}%를 잘라요."
    return vals, "트리밍: 잘라낼 테두리가 없어요."
