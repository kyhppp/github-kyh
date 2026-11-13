"""옛날 사진 복원기 — 웹 UI.

실행: 실행.bat 더블클릭, 또는  .venv\\Scripts\\python.exe app.py
"""
from __future__ import annotations

import shutil
import time
import traceback
import zipfile
from pathlib import Path

import cv2
import gradio as gr
import numpy as np
from PIL import Image, ImageOps

from restorer import auto as AU
from restorer import grading as G
from restorer import pipeline as P
from restorer import trim as T
from restorer.models import use_gpu_default

ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "결과"
OUT_DIR.mkdir(exist_ok=True)

UPSCALE = {"자동 (추천)": 0, "안 함": 1, "2배": 2, "4배": 4}
FACE = {"GFPGAN (자연스러움)": "gfpgan_1.4", "CodeFormer (손상 심한 얼굴에 강함)": "codeformer"}
COLOR = {"기본": "ddcolor", "화사하게": "ddcolor_artistic"}
SR_MODEL = {"디테일 살리기 (추천)": "nomos8k_sc_x4", "매끈하게 (이전 방식)": "real_esrgan"}


# ------------------------------------------------------------------ 변환 도우미

def rgb_to_bgr(a: np.ndarray) -> np.ndarray:
    if a.ndim == 2:
        return cv2.cvtColor(a, cv2.COLOR_GRAY2BGR)
    if a.shape[2] == 4:
        a = a[:, :, :3]
    return a[:, :, ::-1].copy()


def bgr_to_rgb(a: np.ndarray) -> np.ndarray:
    return a[:, :, ::-1].copy()


def editor_to_image_and_mask(ed):
    """ImageEditor 값 → (BGR 이미지, 브러시 마스크)"""
    if ed is None or ed.get("background") is None:
        return None, None
    bg = np.asarray(ed["background"])
    img = rgb_to_bgr(bg)
    mask = np.zeros(img.shape[:2], np.uint8)
    for layer in ed.get("layers") or []:
        layer = np.asarray(layer)
        if layer.ndim == 3 and layer.shape[2] == 4:
            m = (layer[:, :, 3] > 0).astype(np.uint8) * 255
            if m.shape != mask.shape:
                m = cv2.resize(m, (mask.shape[1], mask.shape[0]), interpolation=cv2.INTER_NEAREST)
            mask |= m
    return img, (mask if mask.any() else None)


def make_options(scratch_auto, sensitivity, scratch_method, auto_tone, upscale, upscale_model, face_on, face_model,
                 face_strength, fidelity, colorize, color_model, gpu, user_mask=None) -> P.Options:
    return P.Options(
        scratch_auto=scratch_auto,
        scratch_sensitivity=sensitivity,
        scratch_method="lama" if scratch_method.startswith("AI") else "opencv",
        user_mask=user_mask,
        auto_tone=auto_tone,
        upscale=UPSCALE[upscale],
        upscale_model=SR_MODEL[upscale_model],
        face_restore=face_on,
        face_model=FACE[face_model],
        face_strength=face_strength,
        codeformer_fidelity=fidelity,
        colorize=colorize,
        color_model=COLOR[color_model],
        gpu=gpu,
    )


def save_png(img_bgr: np.ndarray, stem: str) -> Path:
    path = OUT_DIR / f"{stem}_{time.strftime('%Y%m%d_%H%M%S')}.png"
    Image.fromarray(bgr_to_rgb(img_bgr)).save(path, compress_level=3)  # PIL은 한글 경로도 저장 가능, 무손실
    return path


def overlay_mask(img_bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    v = img_bgr.copy()
    v[mask > 0] = (0.35 * v[mask > 0] + 0.65 * np.array([60, 60, 255])).astype(np.uint8)
    return bgr_to_rgb(v)


# ------------------------------------------------------------------ 동작

def preview_mask(ed, sensitivity):
    img, user_mask = editor_to_image_and_mask(ed)
    if img is None:
        raise gr.Error("먼저 사진을 올려 주세요.")
    mask = P.detect_scratches(img, sensitivity)
    if user_mask is not None:
        mask |= user_mask
    pct = (mask > 0).mean() * 100
    return overlay_mask(img, mask), f"빨갛게 표시된 부분({pct:.2f}%)을 메웁니다. 너무 많으면 감도를 낮추세요."


def restore_one(ed, *args, progress=gr.Progress()):
    img, user_mask = editor_to_image_and_mask(ed)
    if img is None:
        raise gr.Error("먼저 사진을 올려 주세요.")
    opt = make_options(*args, user_mask=user_mask)
    t0 = time.time()
    try:
        out = P.restore(img, opt, progress)
    except Exception as e:
        traceback.print_exc()
        raise gr.Error(f"복원 중 오류가 났어요: {e}")
    path = save_png(out, "복원")
    before = cv2.resize(img, (out.shape[1], out.shape[0]), interpolation=cv2.INTER_LANCZOS4)
    info = [f"완료: {time.time() - t0:.1f}초", f"{img.shape[1]}×{img.shape[0]} → {out.shape[1]}×{out.shape[0]}"]
    f = opt.debug.get("upscale", 1.0)
    info.append(f"해상도 {f:.1f}배 높임" if f > 1 else "해상도: 이미 충분히 커서 키우지 않음 (빠르게 처리)")
    if opt.face_restore:
        info.append(f"얼굴 {opt.debug.get('faces', 0)}개 복원")
    m = opt.debug.get("mask")
    if m is not None and m.any():
        info.append(f"긁힘·얼룩 {(m > 0).mean() * 100:.2f}% 메움")
    info.append(f"저장 위치: {path}")
    info.append("아래 '마무리'에서 색감을 다듬고 트리밍(자르기)할 수 있어요.")
    return ((bgr_to_rgb(before), bgr_to_rgb(out)), str(path), "\n".join(info), out, G.thumbnail(out),
            str(path), gr.update(visible=True))


# ------------------------------------------------------------------ 색감 보정

def grade_from_values(vals) -> G.Grade:
    return G.Grade(**dict(zip(G.FIELDS, vals)))


def trim_from_values(vals) -> T.Trim:
    return T.Trim(**dict(zip(T.FIELDS, vals)))


def split_finish(vals):
    """[색감 값들..., 트리밍 값들...] → (Grade, Trim)"""
    n = len(G.FIELDS)
    return grade_from_values(vals[:n]), trim_from_values(vals[n:])


def finish(img: np.ndarray, grade: G.Grade, trim: T.Trim) -> np.ndarray:
    """마무리: 먼저 자르고, 잘린 사진에 색감을 입힌다 (비네팅·색 균형이 새 테두리 기준이 되도록)."""
    return G.apply(T.apply(img, trim), grade)


def grade_preview(small, *vals):
    if small is None:
        return None, None
    grade, trim = split_finish(vals)
    cut = T.apply(small, trim)
    return (bgr_to_rgb(cut), bgr_to_rgb(G.apply(cut, grade))), bgr_to_rgb(T.overlay(small, trim))


def grade_load(img_rgb):
    if img_rgb is None:
        return None, None
    img = rgb_to_bgr(np.asarray(img_rgb))
    return img, G.thumbnail(img)


def grade_save(src, *vals):
    if src is None:
        raise gr.Error("먼저 사진을 복원하거나, '다른 사진 불러와서 마무리하기'로 사진을 올려 주세요.")
    out = finish(src, *split_finish(vals))
    path = save_png(out, "완성")
    return (str(path), f"저장 완료 ({out.shape[1]}×{out.shape[0]})\n저장 위치: {path}",
            str(path), gr.update(visible=True))


# ------------------------------------------------------------------ 삭제

def _remove_in_out_dir(*paths) -> list[str]:
    """결과 폴더 안의 파일·폴더만 지운다. 지운 이름 목록을 돌려준다."""
    removed = []
    base = OUT_DIR.resolve()
    for p in paths:
        if not p:
            continue
        p = Path(p).resolve()
        if base not in p.parents:  # 안전장치: 결과 폴더 밖은 절대 지우지 않음
            continue
        if p.is_dir():
            shutil.rmtree(p)
            removed.append(p.name)
        elif p.exists():
            p.unlink()
            removed.append(p.name)
    return removed


def _hide2():
    return gr.update(visible=False), gr.update(visible=False)


def clear_restore_view():
    """복원 결과 화면 비우기 (저장된 파일은 그대로). 순서: slider, file_out, info, restore_path,
    del_r_btn, del_r_row, mask_view, mask_info"""
    return None, None, "", None, *_hide2(), None, ""


def clear_saved_finish():
    """'저장한 마무리 사진' 표시 비우기 (파일은 그대로). 순서: grade_file, grade_info, grade_path, del_g_btn, del_g_row"""
    return None, "", None, *_hide2()


def clear_finish_view():
    """마무리 칸 전체 비우기. 순서: grade_src, grade_small, grade_view, trim_view, grade_upload + clear_saved_finish"""
    return None, None, None, None, None, *clear_saved_finish()


def delete_restore(path):
    removed = _remove_in_out_dir(path)
    msg = f"🗑 삭제했어요: {', '.join(removed)}" if removed else "삭제할 파일이 없어요 (이미 지워졌을 수 있어요)."
    # 복원 결과와, 그 결과를 이어받은 색감 보정 화면도 비운다
    return (None, None, msg, None, None, None, None, None, *_hide2())


def delete_grade(path):
    removed = _remove_in_out_dir(path)
    msg = f"🗑 삭제했어요: {', '.join(removed)}" if removed else "삭제할 파일이 없어요 (이미 지워졌을 수 있어요)."
    return None, msg, None, *_hide2()


def delete_batch(folder):
    removed = _remove_in_out_dir(folder, folder and str(Path(folder).with_suffix(".zip")))
    msg = f"🗑 삭제했어요: {', '.join(removed)}" if removed else "삭제할 파일이 없어요 (이미 지워졌을 수 있어요)."
    return [], None, msg, None, *_hide2()


def delete_controls(label: str):
    """'삭제' 버튼 + '정말 삭제할까요?' 확인 줄. 처음엔 숨겨 둔다."""
    btn = gr.Button(f"🗑 {label}", variant="stop", size="sm", visible=False)
    with gr.Row(visible=False) as row:
        gr.Markdown("**정말 삭제할까요?** 저장된 파일이 지워지며 되돌릴 수 없어요.")
        yes = gr.Button("정말 삭제", variant="stop", size="sm")
        no = gr.Button("취소", size="sm")
    btn.click(lambda: (gr.update(visible=False), gr.update(visible=True)), None, [btn, row])
    no.click(lambda: (gr.update(visible=True), gr.update(visible=False)), None, [btn, row])
    return btn, row, yes


N_OPTS = 13  # option_panel 이 돌려주는 위젯 수 (make_options 인자 수)


def restore_batch(files, *args, progress=gr.Progress()):
    if not files:
        raise gr.Error("사진 파일을 하나 이상 올려 주세요.")
    opt_vals, finish_vals = args[:N_OPTS], args[N_OPTS:]
    grade, trim = split_finish(finish_vals)
    folder = OUT_DIR / f"일괄_{time.strftime('%Y%m%d_%H%M%S')}"
    folder.mkdir(parents=True)
    results, log = [], []
    for i, f in enumerate(files):
        src = Path(f if isinstance(f, str) else f.name)
        progress(i / len(files), desc=f"{i + 1}/{len(files)}: {src.name}")
        try:
            with Image.open(src) as im:
                # 휴대폰 사진의 회전 정보(EXIF)를 반영 — 한 장 복원 탭(브라우저)과 같은 방향으로
                img = rgb_to_bgr(np.asarray(ImageOps.exif_transpose(im).convert("RGB")))
            opt = make_options(*opt_vals)
            out = finish(P.restore(img, opt), grade, trim)
            dst = folder / f"{src.stem}_복원.png"
            Image.fromarray(bgr_to_rgb(out)).save(dst, compress_level=3)
            results.append(str(dst))
            log.append(f"✔ {src.name}")
        except Exception as e:
            traceback.print_exc()
            log.append(f"✘ {src.name}: {e}")
    zpath = folder.with_suffix(".zip")
    with zipfile.ZipFile(zpath, "w") as z:
        for r in results:
            z.write(r, Path(r).name)
    log.append(f"\n저장 폴더: {folder}")
    return results, str(zpath), "\n".join(log), str(folder), gr.update(visible=True)


# ------------------------------------------------------------------ 화면

LAMA, OPENCV = "AI (LaMa, 자연스러움)", "빠름 (OpenCV)"


def _label(d: dict, value) -> str:
    """값 → 화면에 보이는 선택지 이름 (UPSCALE·FACE 등)"""
    return next(k for k, v in d.items() if v == value)


class Section:
    """설정 묶음 하나: 위젯들, 기본값, '자동' 추천 함수(사진 → (값 목록, 설명))."""

    def __init__(self, widgets, defaults, suggest=None, view="grade"):
        self.widgets, self.defaults, self.suggest, self.view = widgets, defaults, suggest, view
        self.auto_btn = self.reset_btn = None


def section_tools(can_auto: bool):
    """각 설정 칸 맨 위의 [✨ 자동] [↺ 초기화] 버튼"""
    with gr.Row(elem_classes="sec-tools"):
        a = gr.Button("✨ 자동", size="sm", variant="secondary", visible=can_auto, min_width=60)
        r = gr.Button("↺ 초기화", size="sm", variant="secondary", min_width=60)
    return a, r


def panel_tools(can_auto: bool, what: str):
    """패널 맨 위의 [✨ 전체 자동] [↺ 전체 초기화] 버튼"""
    with gr.Row():
        a = gr.Button(f"✨ {what} 전체 자동", variant="secondary", visible=can_auto)
        r = gr.Button(f"↺ {what} 전체 초기화", variant="secondary")
    return a, r


def option_panel(batch: bool = False):
    """복원 옵션 위젯들. (make_options 순서의 위젯 목록, 설정 묶음들, 전체 자동 버튼, 전체 초기화 버튼)"""
    can_auto = not batch  # 여러 장은 사진마다 달라서 '자동'은 한 장 탭에서만
    if can_auto:
        gr.Markdown("**✨ 자동**을 누르면 올린 사진을 분석해 추천값을 넣어 줘요. 확인한 뒤 더 조절하세요.",
                    elem_classes="hint")
    all_auto, all_reset = panel_tools(can_auto, "복원")
    secs = []
    with gr.Accordion("1. 긁힘·얼룩 제거", open=True):
        a, r = section_tools(can_auto)
        scratch_auto = gr.Checkbox(label="자동으로 찾아서 지우기", value=False,
                                   info="얇은 긁힘과 먼지 점을 자동으로 찾습니다." +
                                        ("" if batch else " 직접 칠한 부분은 이 설정과 상관없이 항상 지웁니다."))
        sensitivity = gr.Slider(0, 1, value=0.4, step=0.05, label="자동 감지 감도",
                                info="높을수록 많이 찾지만 실제 무늬까지 지울 수 있어요.")
        scratch_method = gr.Radio([LAMA, OPENCV], value=LAMA, label="메우는 방식")

        def sug_scratch(img):
            d, msg = AU.suggest_scratch(img)
            return [d["on"], d["sensitivity"], LAMA], msg
        secs.append(Section([scratch_auto, sensitivity, scratch_method], [False, 0.4, LAMA], sug_scratch))
        secs[-1].auto_btn, secs[-1].reset_btn = a, r
    with gr.Accordion("2. 색 바램 보정", open=False):
        a, r = section_tools(can_auto)
        auto_tone = gr.Checkbox(label="빛바랜 대비·누런 색조 보정", value=False)

        def sug_tone(img):
            d, msg = AU.suggest_tone(img)
            return [d["on"]], msg
        secs.append(Section([auto_tone], [False], sug_tone))
        secs[-1].auto_btn, secs[-1].reset_btn = a, r
    with gr.Accordion("3. 해상도 높이기", open=True):
        a, r = section_tools(can_auto)
        upscale = gr.Radio(list(UPSCALE), value="자동 (추천)", label="확대 배율",
                           info="자동: 작은 사진만 긴 변 약 4000px까지 키우고, 이미 큰 사진은 그대로 둬서 빠릅니다. "
                                "2배·4배를 골라도 긴 변 8000px을 넘게 키우지는 않아요.")
        upscale_model = gr.Radio(list(SR_MODEL), value=list(SR_MODEL)[0], label="확대 방식",
                                 info="디테일 살리기: 옷감 무늬·머리카락·배경 질감을 살립니다. "
                                      "매끈하게: 잡티가 심한 사진에서 깔끔하지만 그림처럼 뭉개질 수 있어요.")

        def sug_up(img):
            d, msg = AU.suggest_upscale(img)
            return [_label(UPSCALE, d["choice"]), _label(SR_MODEL, d["model"])], msg
        secs.append(Section([upscale, upscale_model], ["자동 (추천)", list(SR_MODEL)[0]], sug_up))
        secs[-1].auto_btn, secs[-1].reset_btn = a, r
    with gr.Accordion("4. 얼굴 복원", open=True):
        a, r = section_tools(can_auto)
        face_on = gr.Checkbox(label="얼굴 또렷하게 복원", value=True)
        face_model = gr.Radio(list(FACE), value=list(FACE)[0], label="얼굴 복원 모델")
        face_strength = gr.Slider(0, 1, value=0.85, step=0.05, label="복원 강도",
                                  info="낮추면 원래 얼굴 느낌이 더 남습니다.")
        fidelity = gr.Slider(0, 1, value=0.7, step=0.05, label="CodeFormer 원본 충실도",
                             info="CodeFormer 전용. 높을수록 원래 얼굴과 닮게, 낮을수록 더 선명하게.")

        def sug_face(img):
            d, msg = AU.suggest_face(img)
            return [d["on"], _label(FACE, d["model"]), d["strength"], d["fidelity"]], msg
        secs.append(Section([face_on, face_model, face_strength, fidelity], [True, list(FACE)[0], 0.85, 0.7], sug_face))
        secs[-1].auto_btn, secs[-1].reset_btn = a, r
    with gr.Accordion("5. 흑백 사진 컬러화", open=True):
        a, r = section_tools(can_auto)
        colorize = gr.Checkbox(label="자동으로 색 입히기", value=False,
                               info="흑백 사진에만 켜세요. 컬러 사진에 켜면 원래 색이 바뀝니다.")
        color_model = gr.Radio(list(COLOR), value="기본", label="색감")

        def sug_color(img):
            d, msg = AU.suggest_colorize(img)
            return [d["on"], _label(COLOR, d["model"])], msg
        secs.append(Section([colorize, color_model], [False, "기본"], sug_color))
        secs[-1].auto_btn, secs[-1].reset_btn = a, r
    with gr.Accordion("기타", open=False):
        gpu = gr.Checkbox(label="그래픽카드(GPU) 가속 사용", value=use_gpu_default(),
                          info="문제가 생기면 끄세요. 끄면 CPU로 처리해 느려집니다.")
    widgets = [scratch_auto, sensitivity, scratch_method, auto_tone, upscale, upscale_model, face_on, face_model,
               face_strength, fidelity, colorize, color_model, gpu]
    assert len(widgets) == N_OPTS
    return widgets, secs, all_auto, all_reset


def finish_panel(batch: bool = False):
    """마무리(색감 보정 + 트리밍) 위젯. 한 장 탭과 여러 장 탭이 똑같이 쓴다.
    (프리셋, 색감 위젯 — Grade 필드 순서, 트리밍 위젯 — Trim 필드 순서, 설정 묶음들, 전체 자동, 전체 초기화)"""
    can_auto = not batch
    all_auto, all_reset = panel_tools(can_auto, "마무리")
    preset = gr.Dropdown(list(G.PRESETS), value="원본 그대로", label="색감 프리셋",
                         info="고르면 아래 슬라이더가 맞춰집니다. 고른 뒤 세부 조절도 가능해요.")
    secs = []

    def grade_sec(widgets, keys, suggest, a, r, view="grade"):
        defaults = [getattr(G.Grade(), k) for k in keys] if view == "grade" else [getattr(T.Trim(), k) for k in keys]

        def sug(img):
            d, msg = suggest(img)
            return [d[k] for k in keys], msg
        sec = Section(widgets, defaults, sug, view)
        sec.auto_btn, sec.reset_btn = a, r
        secs.append(sec)

    with gr.Accordion("빛", open=not batch):
        a, r = section_tools(can_auto)
        exposure = gr.Slider(-100, 100, 0, step=1, label="노출 (전체 밝기)")
        contrast = gr.Slider(-100, 100, 0, step=1, label="대비")
        highlights = gr.Slider(-100, 100, 0, step=1, label="밝은 영역",
                               info="낮추면 하늘·흰옷의 날아간 부분이 살아납니다.")
        shadows = gr.Slider(-100, 100, 0, step=1, label="어두운 영역",
                            info="높이면 그늘진 얼굴이 밝아집니다.")
        grade_sec([exposure, contrast, highlights, shadows], ["exposure", "contrast", "highlights", "shadows"],
                  AU.suggest_light, a, r)
    with gr.Accordion("색", open=not batch):
        a, r = section_tools(can_auto)
        auto_wb = gr.Checkbox(label="자동 색 균형 (누렇거나 푸르게 뜬 색 바로잡기)", value=False)
        temperature = gr.Slider(-100, 100, 0, step=1, label="색온도 (← 차갑게 · 따뜻하게 →)")
        tint = gr.Slider(-100, 100, 0, step=1, label="색조 (← 초록 · 자주 →)")
        saturation = gr.Slider(-100, 100, 0, step=1, label="채도", info="-100이면 흑백이 됩니다.")
        vibrance = gr.Slider(-100, 100, 0, step=1, label="생동감",
                             info="색이 옅은 곳만 살려서 피부색이 덜 과해집니다.")
        grade_sec([auto_wb, temperature, tint, saturation, vibrance],
                  ["auto_wb", "temperature", "tint", "saturation", "vibrance"], AU.suggest_color, a, r)
    with gr.Accordion("효과", open=False):
        a, r = section_tools(can_auto)
        sepia = gr.Slider(0, 100, 0, step=1, label="세피아 톤 (옛날 갈색 사진 느낌)")
        fade = gr.Slider(0, 100, 0, step=1, label="바랜 느낌 (필름 톤)")
        vignette = gr.Slider(0, 100, 0, step=1, label="비네팅 (가장자리 어둡게)")
        sharpness = gr.Slider(0, 100, 0, step=1, label="선명도")
        grade_sec([sepia, fade, vignette, sharpness], ["sepia", "fade", "vignette", "sharpness"],
                  AU.suggest_effects, a, r)
    with gr.Accordion("7. 트리밍 (자르기)", open=True):
        a, r = section_tools(can_auto)
        t_auto = gr.Checkbox(label="스캔 테두리 자동으로 자르기", value=False,
                             info="사진 가장자리의 흰색·검은색 띠(스캔 여백, 인화지 테두리)를 찾아 잘라냅니다." +
                                  (" 사진마다 따로 찾아요." if batch else
                                   " (위 ✨ 자동은 찾은 테두리를 아래 슬라이더에 넣어 줘서 직접 더 조절할 수 있어요.)"))
        t_ratio = gr.Dropdown(list(T.RATIOS), value=T.FREE, label="비율",
                              info="고르면 남은 부분의 가운데를 그 비율로 맞춥니다. 위치는 아래 여백으로 옮기세요.")
        with gr.Row():
            t_top = gr.Slider(0, 45, 0, step=0.5, label="위 (%)")
            t_bottom = gr.Slider(0, 45, 0, step=0.5, label="아래 (%)")
        with gr.Row():
            t_left = gr.Slider(0, 45, 0, step=0.5, label="왼쪽 (%)")
            t_right = gr.Slider(0, 45, 0, step=0.5, label="오른쪽 (%)")
        trim_ctrls = [t_auto, t_top, t_bottom, t_left, t_right, t_ratio]
        grade_sec(trim_ctrls, list(T.FIELDS), AU.suggest_trim, a, r, view="trim")
    grade_ctrls = [auto_wb, exposure, temperature, tint, highlights, shadows, contrast,
                   saturation, vibrance, sepia, fade, vignette, sharpness]
    assert len(grade_ctrls) == len(G.FIELDS) and len(trim_ctrls) == len(T.FIELDS)
    return preset, grade_ctrls, trim_ctrls, secs, all_auto, all_reset


def wire_sections(secs, all_auto, all_reset, img_input, to_image, after=None, preset=None):
    """[✨ 자동]·[↺ 초기화] 버튼 연결.
    img_input: 분석할 사진이 들어 있는 컴포넌트, to_image: 그 값 → BGR 사진 (없으면 None)
    after(view): 값을 바꾼 뒤 이어서 실행할 이벤트 인자들 (미리보기 갱신 등) 목록을 돌려주는 함수
    preset: 마무리 패널의 프리셋 드롭다운 — 자동·초기화로 값이 바뀌면 '원본 그대로'로 돌려 헷갈리지 않게"""

    def chain(ev, view):
        if preset is not None:
            ev = ev.then(lambda: "원본 그대로", None, preset, show_progress="hidden")
        for kw in (after(view) if after else []):
            ev = ev.then(**kw)

    def need_image(v):
        img = to_image(v)
        if img is None:
            raise gr.Error("먼저 사진을 올려 주세요. 자동 설정은 사진을 보고 정해요.")
        return img

    for sec in secs:
        chain(sec.reset_btn.click(lambda d=sec.defaults: list(d), None, sec.widgets, show_progress="hidden"), sec.view)

        def auto_fn(v, sec=sec):
            vals, msg = sec.suggest(need_image(v))
            gr.Info(msg, duration=8)
            return vals
        chain(sec.auto_btn.click(auto_fn, img_input, sec.widgets), sec.view)

    every = [w for sec in secs for w in sec.widgets]
    chain(all_reset.click(lambda: [x for sec in secs for x in sec.defaults], None, every, show_progress="hidden"),
          secs[0].view)

    def all_auto_fn(v):
        img = need_image(v)
        vals, msgs = [], []
        for sec in secs:
            sv, m = sec.suggest(img)
            vals += sv
            msgs.append("• " + m)
        gr.Info("\n".join(msgs), duration=15)
        return vals
    chain(all_auto.click(all_auto_fn, img_input, every), secs[0].view)


def finish_defaults():
    """초기화: 프리셋 + 색감 + 트리밍 기본값"""
    return ["원본 그대로"] + G.preset_values("원본 그대로") + [getattr(T.Trim(), k) for k in T.FIELDS]


VIEW_H = 520  # 왼쪽 사진 보는 칸 높이 (화면 위에 고정되므로 노트북 화면에도 다 들어오게)

CSS = """
.gradio-container {max-width: 1400px !important; margin: auto;
  overflow: clip !important;}  /* hidden 이면 아래 sticky(사진 고정)가 안 먹어서 clip 으로 */
#title {text-align:center; margin-bottom: 0}
#subtitle {text-align:center; color: var(--body-text-color-subdued); margin-top: 4px}
.stage {margin-top: 8px;}
/* 오른쪽 설정을 내려도 왼쪽 사진은 화면에 붙어 있게 (넓은 화면에서만) */
.sec-tools {justify-content: flex-end; gap: 6px; margin-bottom: 2px;}
.sec-tools button {flex: 0 0 auto !important; padding: 2px 12px !important;}
.hint {font-size: 0.92em; color: var(--body-text-color-subdued);}
@media (min-width: 1000px) {
  .viewer {position: sticky; top: 8px; align-self: flex-start; z-index: 5;}
}
"""

with gr.Blocks(title="옛날 사진 복원기") as demo:
    gr.Markdown("# 옛날 사진 복원기", elem_id="title")
    gr.Markdown("긁힘·얼룩 제거 · 해상도 높이기 · 얼굴 복원 · 흑백 컬러화 — 모든 처리는 이 컴퓨터 안에서만 이뤄집니다.",
                elem_id="subtitle")

    with gr.Tab("사진 한 장 복원"):
        # ---------------- 1~5. 복원 — 왼쪽 사진은 고정, 오른쪽 설정만 스크롤
        gr.Markdown("### 1~5. 복원", elem_classes="stage")
        with gr.Row():
            with gr.Column(scale=3, elem_classes="viewer"):
                with gr.Tabs(selected="src") as vtabs:
                    with gr.Tab("📷 원본 · 긁힌 곳 칠하기", id="src"):
                        editor = gr.ImageEditor(
                            label="사진 올리기 (긁힌 곳은 브러시로 칠하면 지워 드려요)", show_label=False,
                            type="numpy", image_mode="RGBA", format="png", height=VIEW_H,
                            sources=["upload", "clipboard"], transforms=(),
                            brush=gr.Brush(colors=["#ff3b3b"], color_mode="fixed", default_size=8),
                            layers=False,
                        )
                    with gr.Tab("✨ 복원 전 / 후", id="result"):
                        slider = gr.ImageSlider(label="복원 전 / 후 (가운데 막대를 끌어 비교)", max_height=VIEW_H,
                                                buttons=["fullscreen"])
                    with gr.Tab("🔍 긁힘 감지", id="mask"):
                        mask_view = gr.Image(label="긁힘·얼룩으로 감지된 부분 (빨강)", height=VIEW_H - 40,
                                             interactive=False)
                        mask_info = gr.Markdown()
                with gr.Row():
                    btn_mask = gr.Button("🔍 자동 감지 미리보기", variant="secondary")
                    btn_run = gr.Button("복원 시작", variant="primary", size="lg", scale=2)
            with gr.Column(scale=2):
                opts, osecs, o_all_auto, o_all_reset = option_panel()
                with gr.Group():
                    info = gr.Textbox(label="결과 정보", lines=4)
                    file_out = gr.File(label="복원된 사진 내려받기 (PNG)", height=90)
                restore_path = gr.State()
                del_r_btn, del_r_row, del_r_yes = delete_controls("결과가 마음에 안 들면 삭제")

        # ---------------- 6~7. 마무리 (복원 후)
        grade_src = gr.State()    # 원본 해상도 (BGR)
        grade_small = gr.State()  # 미리보기용 축소본 (BGR)
        gr.Markdown("### 6~7. 마무리: 색감 보정 · 트리밍\n"
                    "복원이 끝나면 결과가 여기로 자동으로 넘어옵니다. 슬라이더를 움직이면 바로 미리보기가 바뀌고, "
                    "**마무리 적용해서 저장**을 누르면 원래 해상도로 저장됩니다.", elem_classes="stage")
        with gr.Row():
            with gr.Column(scale=3, elem_classes="viewer"):
                with gr.Tabs(selected="grade") as ftabs:
                    with gr.Tab("🎨 색감 전 / 후", id="grade"):
                        grade_view = gr.ImageSlider(label="색감 보정 전 / 후 — 트리밍한 모양 (가운데 막대를 끌어 비교)",
                                                    max_height=VIEW_H, buttons=["fullscreen"])
                    with gr.Tab("✂ 트리밍 범위", id="trim"):
                        trim_view = gr.Image(label="밝은 부분만 남고, 어두운 부분은 잘려 나가요",
                                             height=VIEW_H, interactive=False)
                with gr.Row():
                    btn_grade_clear = gr.Button("🧹 사진 비우기", variant="secondary")
                    btn_grade_reset = gr.Button("설정 초기화", variant="secondary")
                    btn_grade_save = gr.Button("마무리 적용해서 저장", variant="primary", size="lg", scale=2)
            with gr.Column(scale=2):
                with gr.Accordion("다른 사진 불러와서 마무리(색감·트리밍)만 하기", open=False):
                    grade_upload = gr.Image(label="사진 올리기", type="numpy", image_mode="RGB",
                                            sources=["upload", "clipboard"], height=200)
                preset, grade_ctrls, trim_ctrls, fsecs, f_all_auto, f_all_reset = finish_panel()
                with gr.Group():
                    grade_file = gr.File(label="마무리한 사진 내려받기 (PNG)", height=90)
                    grade_info = gr.Markdown()
                grade_path = gr.State()
                del_g_btn, del_g_row, del_g_yes = delete_controls("저장한 마무리 사진 삭제")

        finish_ctrls = grade_ctrls + trim_ctrls
        preview_args = dict(fn=grade_preview, inputs=[grade_small] + finish_ctrls,
                            outputs=[grade_view, trim_view], show_progress="hidden")

        def show(tabs, tab_id):
            """보는 탭 바꾸기 (예: 복원이 끝나면 '복원 전 / 후' 탭으로)"""
            return dict(fn=lambda: gr.Tabs(selected=tab_id), inputs=None, outputs=tabs, show_progress="hidden")

        restore_view = [slider, file_out, info, restore_path, del_r_btn, del_r_row, mask_view, mask_info]
        saved_finish = [grade_file, grade_info, grade_path, del_g_btn, del_g_row]
        finish_view = [grade_src, grade_small, grade_view, trim_view, grade_upload] + saved_finish

        # 원본 사진을 지우거나 새 사진을 올리면, 이전 사진의 복원 결과·마무리 칸도 함께 비운다
        for ev in (editor.clear, editor.upload):
            ev(clear_restore_view, None, restore_view, show_progress="hidden") \
                .then(clear_finish_view, None, finish_view, show_progress="hidden")
        btn_grade_clear.click(clear_finish_view, None, finish_view, show_progress="hidden")
        grade_upload.clear(clear_finish_view, None, finish_view, show_progress="hidden")

        btn_mask.click(preview_mask, [editor, opts[1]], [mask_view, mask_info]).success(**show(vtabs, "mask"))
        # 새로 복원하면 마무리 칸은 새 결과로 바뀌므로, 따로 불러온 사진·이전에 저장한 표시는 비운다
        btn_run.click(restore_one, [editor] + opts,
                      [slider, file_out, info, grade_src, grade_small, restore_path, del_r_btn]) \
            .success(**show(vtabs, "result")) \
            .then(lambda: (None, *clear_saved_finish()), None, [grade_upload] + saved_finish, show_progress="hidden") \
            .then(**preview_args)
        del_r_yes.click(delete_restore, restore_path,
                        [slider, file_out, info, grade_src, grade_small, grade_view, trim_view, restore_path,
                         del_r_btn, del_r_row]).then(**show(vtabs, "src")) \
            .then(lambda: (None, *clear_saved_finish()), None, [grade_upload] + saved_finish, show_progress="hidden")
        del_g_yes.click(delete_grade, grade_path, [grade_file, grade_info, grade_path, del_g_btn, del_g_row])
        for c in finish_ctrls:
            c.input(trigger_mode="always_last", **preview_args)
        # 만지는 슬라이더에 맞춰 왼쪽 미리보기 탭도 바꿔 준다
        for c in grade_ctrls + [preset]:
            c.input(**show(ftabs, "grade"))
        for c in trim_ctrls:
            c.input(**show(ftabs, "trim"))
        preset.input(G.preset_values, preset, grade_ctrls).then(**preview_args)
        btn_grade_reset.click(finish_defaults, None, [preset] + finish_ctrls).then(**preview_args)
        grade_upload.upload(grade_load, grade_upload, [grade_src, grade_small]) \
            .then(clear_saved_finish, None, saved_finish, show_progress="hidden").then(**preview_args)
        btn_grade_save.click(grade_save, [grade_src] + finish_ctrls, [grade_file, grade_info, grade_path, del_g_btn])

        # [✨ 자동]·[↺ 초기화] — 복원 설정은 올린 원본을, 마무리는 복원 결과(미리보기본)를 보고 정한다
        wire_sections(osecs, o_all_auto, o_all_reset, editor, lambda ed: editor_to_image_and_mask(ed)[0])
        wire_sections(fsecs, f_all_auto, f_all_reset, grade_small, lambda img: img,
                      after=lambda view: [preview_args, show(ftabs, view)], preset=preset)

    with gr.Tab("여러 장 한꺼번에"):
        gr.Markdown("여러 사진을 같은 설정으로 한 번에 복원합니다. 복원 옵션, 색감 보정, 트리밍 모두 "
                    "**사진 한 장 복원과 똑같이** 적용돼요. 결과는 `결과` 폴더와 ZIP 파일로 저장됩니다.")
        with gr.Row():
            with gr.Column(scale=3, elem_classes="viewer"):
                files = gr.File(label="사진 파일들", file_count="multiple", file_types=["image"], height=150)
                btn_batch = gr.Button("모두 복원", variant="primary", size="lg")
                gallery = gr.Gallery(label="결과", columns=4, height=VIEW_H - 120)
            with gr.Column(scale=2):
                btn_copy = gr.Button("⇐ '사진 한 장 복원' 탭 설정 그대로 가져오기", variant="secondary")
                gr.Markdown("한 장으로 먼저 시험해 보고 마음에 드는 설정을 찾은 뒤 누르면, 복원 옵션·색감·트리밍이 "
                            "모두 이쪽으로 복사돼요. (브러시로 칠한 긁힘은 사진마다 달라서 복사되지 않아요.)")
                bopts, bosecs, bo_all_auto, bo_all_reset = option_panel(batch=True)
                with gr.Accordion("6~7. 마무리 (색감 보정 · 트리밍)", open=True):
                    bpreset, bgrade_ctrls, btrim_ctrls, bfsecs, bf_all_auto, bf_all_reset = finish_panel(batch=True)
                    btn_breset = gr.Button("마무리 초기화", variant="secondary", size="sm")
                zip_out = gr.File(label="전체 결과 ZIP", height=90)
                blog = gr.Textbox(label="처리 기록", lines=8)
                batch_folder = gr.State()
                del_b_btn, del_b_row, del_b_yes = delete_controls("이번 결과 전체 삭제 (폴더 + ZIP)")
        bfinish_ctrls = bgrade_ctrls + btrim_ctrls
        wire_sections(bosecs, bo_all_auto, bo_all_reset, files, lambda _v: None)
        wire_sections(bfsecs, bf_all_auto, bf_all_reset, files, lambda _v: None, preset=bpreset)
        bpreset.input(G.preset_values, bpreset, bgrade_ctrls)
        btn_breset.click(finish_defaults, None, [bpreset] + bfinish_ctrls)
        btn_batch.click(restore_batch, [files] + bopts + bfinish_ctrls,
                        [gallery, zip_out, blog, batch_folder, del_b_btn])
        del_b_yes.click(delete_batch, batch_folder, [gallery, zip_out, blog, batch_folder, del_b_btn, del_b_row])

    # 한 장 탭 설정 → 여러 장 탭 (복원 옵션 + 프리셋 + 색감 + 트리밍, 순서가 서로 같음)
    btn_copy.click(lambda *v: list(v), opts + [preset] + finish_ctrls, bopts + [bpreset] + bfinish_ctrls) \
        .then(lambda: gr.Info("'사진 한 장 복원' 탭의 설정을 가져왔어요."), None, None)

    with gr.Tab("사용 방법"):
        gr.Markdown("""
### 추천 설정
| 사진 상태 | 추천 |
|---|---|
| 작고 흐릿한 인물 사진 | 해상도 2~4배 + 얼굴 복원 |
| 흑백 옛날 사진 | 위 설정 + **컬러화** |
| 긁힘·먼지가 많은 사진 | **자동 감지 미리보기**로 확인 후 감도 조절, 놓친 곳은 브러시로 칠하기 |
| 누렇게 바랜 컬러 사진 | **색 바램 보정** |
| 얼굴이 심하게 뭉개진 사진 | 얼굴 모델을 **CodeFormer**로, 충실도 0.5 전후 |
| 컬러화 색이 어색할 때 | 복원 후 **색감 보정**에서 *자동 색 균형*, *색온도*, *채도* 조절 |
| 복원 결과가 밋밋할 때 | **색감 보정** 프리셋 *자연스럽게 다듬기* 또는 *선명하고 화사하게* |
| 스캔한 사진에 흰 테두리·검은 띠가 있을 때 | **트리밍**의 *스캔 테두리 자동으로 자르기* |
| 액자·인화 크기에 맞추고 싶을 때 | **트리밍**에서 비율(예: *3:2 가로*)을 고르고, 위·아래·왼쪽·오른쪽 여백으로 위치 조절 |
| 여러 장을 같은 느낌으로 | 한 장으로 먼저 설정을 찾고 → **여러 장** 탭의 *설정 그대로 가져오기* |

### 알아두면 좋은 점
- 처음 각 기능을 쓸 때 AI 모델을 내려받습니다 (합계 약 2GB, 한 번만).
- 얼굴 복원 AI는 흐릿한 부분을 **그럴듯하게 새로 그려내는** 방식이라, 실제와 조금 다른 얼굴이 될 수 있어요. 어색하면 *복원 강도*를 낮추세요.
- 컬러화된 색은 AI가 추측한 색이며 실제 색과 다를 수 있어요.
- 자동 긁힘 감지는 머리카락·벽돌 무늬처럼 가는 선을 긁힘으로 착각할 수 있어요. 미리보기로 확인하고, 확실한 긁힘은 브러시로 칠하는 게 가장 정확합니다.
- 결과 파일은 프로그램 폴더 안의 `결과` 폴더에도 저장됩니다.
""")


if __name__ == "__main__":
    import secrets
    import sys

    share = "--share" in sys.argv  # 외부에서 접속 가능한 임시 공개 링크 생성
    auth = None
    if share:
        # 공개 링크에는 비밀번호를 건다. 비밀번호.txt 첫 줄이 비밀번호 (없으면 새로 만듦)
        pw_file = ROOT / "비밀번호.txt"
        if not pw_file.exists() or not pw_file.read_text(encoding="utf-8").strip():
            pw_file.write_text(secrets.token_urlsafe(9) + "\n", encoding="utf-8")
        password = pw_file.read_text(encoding="utf-8").splitlines()[0].strip()
        auth = lambda _user, pw: secrets.compare_digest(pw, password)  # noqa: E731
        print(f"\n  접속 비밀번호: {password}   (아이디는 아무거나 입력)\n  바꾸려면 비밀번호.txt 를 고친 뒤 다시 실행하세요.\n")
    demo.queue(default_concurrency_limit=1).launch(
        inbrowser=True, share=share, auth=auth,
        auth_message="옛날 사진 복원기 — 아이디는 아무거나, 비밀번호를 입력하세요.",
        css=CSS, theme=gr.themes.Soft())
