"""마무리 트리밍(자르기).

모든 이미지는 BGR uint8. 여백은 사진 크기에 대한 퍼센트라서
미리보기용 축소본과 원본 해상도에 똑같이 적용된다.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import cv2
import numpy as np

# 비율 이름 → (가로, 세로). None 이면 비율을 맞추지 않고 여백만 자른다.
RATIOS: dict[str, tuple[int, int] | None] = {
    "자유 (여백만 자르기)": None,
    "1:1 정사각형": (1, 1),
    "4:3 가로": (4, 3),
    "3:4 세로": (3, 4),
    "3:2 가로 (사진 인화)": (3, 2),
    "2:3 세로 (사진 인화)": (2, 3),
    "16:9 가로 (화면)": (16, 9),
    "4:5 세로 (SNS)": (4, 5),
}
FREE = next(iter(RATIOS))


@dataclass
class Trim:
    auto_border: bool = False  # 스캔할 때 생긴 흰색·검은색 테두리를 자동으로 잘라냄
    top: float = 0             # 위 여백 % (0~45)
    bottom: float = 0          # 아래 여백 %
    left: float = 0            # 왼쪽 여백 %
    right: float = 0           # 오른쪽 여백 %
    ratio: str = FREE          # RATIOS 의 이름

    def is_identity(self) -> bool:
        return self == Trim()


FIELDS = list(asdict(Trim()).keys())


def _scan(lines: np.ndarray, limit: int, tol: float) -> int:
    """앞에서부터 '테두리 줄'이 몇 줄 이어지는지. 줄의 97% 이상이 같은 밝기이고,
    그 밝기가 아주 밝거나(흰 여백) 아주 어두우면(검은 띠) 테두리로 본다 — 하늘·벽 같은 실제 배경은 남김."""
    k, ref = 0, None
    for i in range(min(limit, len(lines))):
        row = lines[i]
        med = float(np.median(row))
        if 45 < med < 210 or (np.abs(row - med) <= tol).mean() < 0.97:
            break
        if ref is None:
            ref = med
        elif abs(med - ref) > tol * 2:
            break
        k = i + 1
    return k


def detect_border(img: np.ndarray, max_pct: float = 25, tol: float = 12) -> tuple[float, float, float, float]:
    """가장자리의 균일한 흰색·검은색 띠(스캔 테두리, 인화지 여백)를 찾아 (위, 아래, 왼쪽, 오른쪽) % 로 돌려준다.
    모서리에서 위쪽 흰 여백과 옆쪽 검은 띠가 겹칠 수 있어서, 찾은 만큼 잘라 가며 몇 번 되풀이한다."""
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    s = 600 / max(g.shape)
    if s < 1:
        g = cv2.resize(g, (round(g.shape[1] * s), round(g.shape[0] * s)), interpolation=cv2.INTER_AREA)
    g = cv2.GaussianBlur(g.astype(np.float32), (0, 0), 1.0)
    H, W = g.shape
    t = b = l = r = 0
    for _ in range(4):
        sub = g[t:H - b, l:W - r]
        if sub.shape[0] < 16 or sub.shape[1] < 16:
            break
        dt = _scan(sub, int(H * max_pct / 100) - t, tol)
        db = _scan(sub[::-1], int(H * max_pct / 100) - b, tol)
        dl = _scan(sub.T, int(W * max_pct / 100) - l, tol)
        dr = _scan(sub.T[::-1], int(W * max_pct / 100) - r, tol)
        if not (dt or db or dl or dr):
            break
        t, b, l, r = t + dt, b + db, l + dl, r + dr

    def pct(k: int, n: int) -> float:
        return 0.0 if k == 0 else min(max_pct, (k + 2) / n * 100)  # 경계의 번진 줄까지 함께 자름

    return pct(t, H), pct(b, H), pct(l, W), pct(r, W)


def crop_box(h: int, w: int, t: Trim, border=(0.0, 0.0, 0.0, 0.0)) -> tuple[int, int, int, int]:
    """(y0, y1, x0, x1). 자동 테두리와 직접 정한 여백을 더한 뒤, 고른 비율로 가운데를 맞춘다."""
    top, bottom = t.top + border[0], t.bottom + border[1]
    left, right = t.left + border[2], t.right + border[3]
    # 한 축에서 90% 넘게 잘라내지 않도록
    if top + bottom > 90:
        k = 90 / (top + bottom)
        top, bottom = top * k, bottom * k
    if left + right > 90:
        k = 90 / (left + right)
        left, right = left * k, right * k
    y0, y1 = round(h * top / 100), h - round(h * bottom / 100)
    x0, x1 = round(w * left / 100), w - round(w * right / 100)
    if y1 - y0 < 8 or x1 - x0 < 8:
        return 0, h, 0, w

    r = RATIOS.get(t.ratio)
    if r:
        target = r[0] / r[1]
        cw, ch = x1 - x0, y1 - y0
        if cw / ch > target:
            nw = max(1, round(ch * target))
            x0 += (cw - nw) // 2
            x1 = x0 + nw
        else:
            nh = max(1, round(cw / target))
            y0 += (ch - nh) // 2
            y1 = y0 + nh
    return y0, y1, x0, x1


def _box(img: np.ndarray, t: Trim) -> tuple[int, int, int, int]:
    border = detect_border(img) if t.auto_border else (0.0, 0.0, 0.0, 0.0)
    return crop_box(img.shape[0], img.shape[1], t, border)


def apply(img: np.ndarray, t: Trim) -> np.ndarray:
    if t.is_identity():
        return img
    y0, y1, x0, x1 = _box(img, t)
    return np.ascontiguousarray(img[y0:y1, x0:x1])


def overlay(img: np.ndarray, t: Trim) -> np.ndarray:
    """잘려 나갈 부분을 어둡게, 남는 부분에 테두리를 그린 미리보기 (BGR)"""
    v = img.copy()
    if t.is_identity():
        return v
    y0, y1, x0, x1 = _box(img, t)
    keep = np.zeros(img.shape[:2], bool)
    keep[y0:y1, x0:x1] = True
    v[~keep] = (v[~keep] * 0.35).astype(np.uint8)
    th = max(2, max(img.shape[:2]) // 300)
    cv2.rectangle(v, (x0, y0), (x1 - 1, y1 - 1), (60, 200, 255), th)
    return v
