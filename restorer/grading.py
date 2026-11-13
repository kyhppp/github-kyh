"""복원 후 색감 보정 (AI 없이 즉시 계산).

모든 이미지는 BGR uint8. 슬라이더 값은 대부분 -100~100 (0 = 변화 없음).
처리 순서: 자동 색 균형 → 노출 → 색온도·색조 → 밝은/어두운 영역 → 대비
          → 채도·생동감 → 세피아 → 바랜 느낌 → 비네팅 → 선명도
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import cv2
import numpy as np

_LUMA = np.array([0.114, 0.587, 0.299], np.float32)  # BGR 순서


@dataclass
class Grade:
    auto_wb: bool = False      # 회색 기준 자동 색 균형
    exposure: float = 0        # 노출 -100~100 (±1.5 스톱)
    temperature: float = 0     # 색온도 -100(차갑게)~100(따뜻하게)
    tint: float = 0            # 색조 -100(초록)~100(자주)
    highlights: float = 0      # 밝은 영역 -100(누르기)~100(밝히기)
    shadows: float = 0         # 어두운 영역 -100(누르기)~100(밝히기)
    contrast: float = 0        # 대비 -100~100
    saturation: float = 0      # 채도 -100(흑백)~100
    vibrance: float = 0        # 생동감 -100~100 (색이 옅은 곳 위주)
    sepia: float = 0           # 세피아 톤 0~100
    fade: float = 0            # 바랜 느낌(검정 들뜸) 0~100
    vignette: float = 0        # 가장자리 어둡게 0~100
    sharpness: float = 0       # 선명도 0~100

    def is_identity(self) -> bool:
        return self == Grade()


FIELDS = list(asdict(Grade()).keys())

# 프리셋: 지정하지 않은 값은 0
PRESETS: dict[str, dict] = {
    "원본 그대로": {},
    "자연스럽게 다듬기": dict(auto_wb=True, contrast=10, vibrance=20, shadows=15, sharpness=20),
    "따뜻한 햇살": dict(temperature=35, tint=5, vibrance=15, shadows=10, contrast=5),
    "시원하고 맑게": dict(temperature=-30, exposure=5, contrast=10, vibrance=15, highlights=-10),
    "선명하고 화사하게": dict(contrast=20, saturation=15, vibrance=25, sharpness=35, shadows=10),
    "필름 카메라": dict(temperature=12, contrast=-10, fade=35, saturation=-10, vignette=25, tint=-5),
    "빈티지 세피아": dict(sepia=75, contrast=10, fade=25, vignette=40),
    "흑백 (깊이 있게)": dict(saturation=-100, contrast=25, shadows=-10, highlights=-15, vignette=20),
    "부드러운 파스텔": dict(contrast=-25, fade=20, saturation=-15, exposure=8, highlights=-20),
}


def preset_values(name: str) -> list:
    """프리셋 → FIELDS 순서의 값 목록"""
    base = asdict(Grade())
    base.update(PRESETS.get(name, {}))
    return [base[k] for k in FIELDS]


def _luma(x: np.ndarray) -> np.ndarray:
    return x @ _LUMA


def apply(img: np.ndarray, g: Grade) -> np.ndarray:
    if g.is_identity():
        return img
    x = img.astype(np.float32) / 255.0
    h, w = x.shape[:2]
    long_side = max(h, w)

    # 자동 색 균형: 밝은 쪽 평균을 회색으로 (회색 세계 가정, 강도 70%)
    if g.auto_wb:
        L = _luma(x)
        sel = (L > np.percentile(L, 20)) & (L < np.percentile(L, 99))
        if sel.sum() > 100:
            mean = x[sel].mean(axis=0)
            gain = mean.mean() / np.maximum(mean, 1e-4)
            x *= 1 + (np.clip(gain, 0.6, 1.6) - 1) * 0.7

    if g.exposure:
        x *= 2.0 ** (g.exposure / 100 * 1.5)

    if g.temperature or g.tint:
        t, ti = g.temperature / 100, g.tint / 100
        x *= np.array([1 - 0.25 * t, 1 - 0.15 * ti, 1 + 0.25 * t], np.float32)  # B, G, R

    if g.highlights or g.shadows:
        L = np.clip(_luma(x), 0, 1)
        # 주변 밝기를 기준으로 해서 질감이 뭉개지지 않게
        base = cv2.GaussianBlur(L, (0, 0), max(2.0, long_side / 150))
        delta = (g.shadows / 100) * 0.45 * (1 - base) ** 3 \
            + (g.highlights / 100) * 0.35 * base ** 3
        ratio = np.clip(L + delta, 0, 1.2) / np.maximum(L, 1e-3)
        x *= ratio[:, :, None]

    if g.contrast:
        c = g.contrast / 100
        mid = float(np.clip(_luma(x).mean(), 0.2, 0.8))
        x = (x - mid) * (1 + c * (0.8 if c > 0 else 0.6)) + mid

    if g.saturation or g.vibrance:
        L = _luma(x)[:, :, None]
        factor = 1 + g.saturation / 100
        if g.vibrance:
            sat = np.clip(x.max(axis=2) - x.min(axis=2), 0, 1)[:, :, None]
            factor = factor * (1 + g.vibrance / 100 * (1 - sat) ** 2)
        x = L + (x - L) * factor

    if g.sepia:
        s = g.sepia / 100
        L = np.clip(_luma(x), 0, 1)[:, :, None]
        tone = L * np.array([0.78, 0.98, 1.12], np.float32) + np.array([0.0, 0.01, 0.03], np.float32)
        x = x * (1 - s) + tone * s

    if g.fade:
        f = g.fade / 100
        x = x * (1 - 0.2 * f) + 0.12 * f  # 검정이 회색빛으로 뜨고, 흰색은 살짝 눌림

    if g.vignette:
        yy, xx = np.ogrid[:h, :w]
        r = np.sqrt(((xx - w / 2) / (w / 2)) ** 2 + ((yy - h / 2) / (h / 2)) ** 2) / np.sqrt(2)
        v = 1 - (g.vignette / 100) * 0.6 * np.clip((r - 0.35) / 0.65, 0, 1) ** 2
        x *= v.astype(np.float32)[:, :, None]

    x = np.clip(x, 0, 1)

    if g.sharpness:
        # 반경을 사진 크기에 비례시켜 미리보기와 원본의 느낌을 맞춤
        sigma = max(0.8, long_side / 1200)
        blur = cv2.GaussianBlur(x, (0, 0), sigma)
        x = np.clip(x + (x - blur) * (g.sharpness / 100) * 1.5, 0, 1)

    return (x * 255 + 0.5).astype(np.uint8)


def thumbnail(img: np.ndarray, max_side: int = 1100) -> np.ndarray:
    """빠른 미리보기용 축소본"""
    h, w = img.shape[:2]
    s = max_side / max(h, w)
    if s >= 1:
        return img
    return cv2.resize(img, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)
