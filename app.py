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
from PIL import Image

from restorer import grading as G
from restorer import pipeline as P
from restorer.models import use_gpu_default

ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "결과"
OUT_DIR.mkdir(exist_ok=True)

UPSCALE = {"안 함": 1, "2배": 2, "4배": 4}
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
    Image.fromarray(bgr_to_rgb(img_bgr)).save(path)  # PIL은 한글 경로도 저장 가능
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
    if opt.face_restore:
        info.append(f"얼굴 {opt.debug.get('faces', 0)}개 복원")
    m = opt.debug.get("mask")
    if m is not None and m.any():
        info.append(f"긁힘·얼룩 {(m > 0).mean() * 100:.2f}% 메움")
    info.append(f"저장 위치: {path}")
    info.append("아래 '색감 보정'에서 색감을 더 다듬을 수 있어요.")
    return ((bgr_to_rgb(before), bgr_to_rgb(out)), str(path), "\n".join(info), out, G.thumbnail(out),
            str(path), gr.update(visible=True))


# ------------------------------------------------------------------ 색감 보정

def grade_from_values(vals) -> G.Grade:
    return G.Grade(**dict(zip(G.FIELDS, vals)))


def grade_preview(small, *vals):
    if small is None:
        return None
    return bgr_to_rgb(small), bgr_to_rgb(G.apply(small, grade_from_values(vals)))


def grade_load(img_rgb):
    if img_rgb is None:
        return None, None
    img = rgb_to_bgr(np.asarray(img_rgb))
    return img, G.thumbnail(img)


def grade_save(src, *vals):
    if src is None:
        raise gr.Error("먼저 사진을 복원하거나, '다른 사진 불러오기'로 사진을 올려 주세요.")
    out = G.apply(src, grade_from_values(vals))
    path = save_png(out, "색감")
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


def delete_restore(path):
    removed = _remove_in_out_dir(path)
    msg = f"🗑 삭제했어요: {', '.join(removed)}" if removed else "삭제할 파일이 없어요 (이미 지워졌을 수 있어요)."
    # 복원 결과와, 그 결과를 이어받은 색감 보정 화면도 비운다
    return (None, None, msg, None, None, None, None, *_hide2())


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


def restore_batch(files, preset, *args, progress=gr.Progress()):
    if not files:
        raise gr.Error("사진 파일을 하나 이상 올려 주세요.")
    grade = grade_from_values(G.preset_values(preset))
    folder = OUT_DIR / f"일괄_{time.strftime('%Y%m%d_%H%M%S')}"
    folder.mkdir(parents=True)
    results, log = [], []
    for i, f in enumerate(files):
        src = Path(f if isinstance(f, str) else f.name)
        progress(i / len(files), desc=f"{i + 1}/{len(files)}: {src.name}")
        try:
            img = rgb_to_bgr(np.asarray(Image.open(src).convert("RGB")))
            opt = make_options(*args)
            out = G.apply(P.restore(img, opt), grade)
            dst = folder / f"{src.stem}_복원.png"
            Image.fromarray(bgr_to_rgb(out)).save(dst)
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

def option_panel(batch: bool = False):
    """복원 옵션 위젯들을 만들고 make_options 순서대로 돌려준다."""
    with gr.Accordion("1. 긁힘·얼룩 제거", open=True):
        scratch_auto = gr.Checkbox(label="자동으로 찾아서 지우기", value=False,
                                   info="얇은 긁힘과 먼지 점을 자동으로 찾습니다." +
                                        ("" if batch else " 직접 칠한 부분은 이 설정과 상관없이 항상 지웁니다."))
        sensitivity = gr.Slider(0, 1, value=0.4, step=0.05, label="자동 감지 감도",
                                info="높을수록 많이 찾지만 실제 무늬까지 지울 수 있어요.")
        scratch_method = gr.Radio(["AI (LaMa, 자연스러움)", "빠름 (OpenCV)"], value="AI (LaMa, 자연스러움)",
                                  label="메우는 방식")
    with gr.Accordion("2. 색 바램 보정", open=False):
        auto_tone = gr.Checkbox(label="빛바랜 대비·누런 색조 보정", value=False)
    with gr.Accordion("3. 해상도 높이기", open=True):
        upscale = gr.Radio(list(UPSCALE), value="2배", label="확대 배율",
                           info="큰 사진을 4배로 키우면 시간이 오래 걸리고 파일이 매우 커집니다.")
        upscale_model = gr.Radio(list(SR_MODEL), value=list(SR_MODEL)[0], label="확대 방식",
                                 info="디테일 살리기: 옷감 무늬·머리카락·배경 질감을 살립니다. "
                                      "매끈하게: 잡티가 심한 사진에서 깔끔하지만 그림처럼 뭉개질 수 있어요.")
    with gr.Accordion("4. 얼굴 복원", open=True):
        face_on = gr.Checkbox(label="얼굴 또렷하게 복원", value=True)
        face_model = gr.Radio(list(FACE), value=list(FACE)[0], label="얼굴 복원 모델")
        face_strength = gr.Slider(0, 1, value=0.85, step=0.05, label="복원 강도",
                                  info="낮추면 원래 얼굴 느낌이 더 남습니다.")
        fidelity = gr.Slider(0, 1, value=0.7, step=0.05, label="CodeFormer 원본 충실도",
                             info="CodeFormer 전용. 높을수록 원래 얼굴과 닮게, 낮을수록 더 선명하게.")
    with gr.Accordion("5. 흑백 사진 컬러화", open=True):
        colorize = gr.Checkbox(label="자동으로 색 입히기", value=False,
                               info="흑백 사진에만 켜세요. 컬러 사진에 켜면 원래 색이 바뀝니다.")
        color_model = gr.Radio(list(COLOR), value="기본", label="색감")
    with gr.Accordion("기타", open=False):
        gpu = gr.Checkbox(label="그래픽카드(GPU) 가속 사용", value=use_gpu_default(),
                          info="문제가 생기면 끄세요. 끄면 CPU로 처리해 느려집니다.")
    return [scratch_auto, sensitivity, scratch_method, auto_tone, upscale, upscale_model, face_on, face_model,
            face_strength, fidelity, colorize, color_model, gpu]


CSS = """
.gradio-container {max-width: 1400px !important; margin: auto;}
#title {text-align:center; margin-bottom: 0}
#subtitle {text-align:center; color: var(--body-text-color-subdued); margin-top: 4px}
"""

with gr.Blocks(title="옛날 사진 복원기") as demo:
    gr.Markdown("# 옛날 사진 복원기", elem_id="title")
    gr.Markdown("긁힘·얼룩 제거 · 해상도 높이기 · 얼굴 복원 · 흑백 컬러화 — 모든 처리는 이 컴퓨터 안에서만 이뤄집니다.",
                elem_id="subtitle")

    with gr.Tab("사진 한 장 복원"):
        with gr.Row():
            with gr.Column(scale=3):
                editor = gr.ImageEditor(
                    label="사진 올리기 (긁힌 곳은 브러시로 칠하면 지워 드려요)",
                    type="numpy", image_mode="RGBA", format="png", height=560,
                    sources=["upload", "clipboard"], transforms=(),
                    brush=gr.Brush(colors=["#ff3b3b"], color_mode="fixed", default_size=8),
                    layers=False,
                )
                with gr.Row():
                    btn_mask = gr.Button("자동 감지 결과 미리보기", variant="secondary")
                    btn_run = gr.Button("복원 시작", variant="primary", size="lg")
            with gr.Column(scale=2):
                opts = option_panel()
        with gr.Row():
            with gr.Column(scale=3):
                slider = gr.ImageSlider(label="복원 전 / 후 (가운데 막대를 끌어 비교)", max_height=720,
                                        buttons=["fullscreen"])
            with gr.Column(scale=2):
                info = gr.Textbox(label="결과 정보", lines=6)
                file_out = gr.File(label="복원된 사진 내려받기 (PNG)")
                restore_path = gr.State()
                del_r_btn, del_r_row, del_r_yes = delete_controls("결과가 마음에 안 들면 삭제")
                mask_view = gr.Image(label="긁힘·얼룩으로 감지된 부분 (빨강)", visible=True, height=300)
                mask_info = gr.Markdown()

        # ---------------- 6. 색감 보정 (복원 후)
        grade_src = gr.State()    # 원본 해상도 (BGR)
        grade_small = gr.State()  # 미리보기용 축소본 (BGR)
        gr.Markdown("---\n## 6. 색감 보정 (복원 후)\n"
                    "복원이 끝나면 결과가 여기로 자동으로 넘어옵니다. 슬라이더를 움직이면 바로 미리보기가 바뀌고, "
                    "**색감 적용해서 저장**을 누르면 원래 해상도로 저장됩니다.")
        with gr.Row():
            with gr.Column(scale=3):
                grade_view = gr.ImageSlider(label="색감 보정 전 / 후 (가운데 막대를 끌어 비교)", max_height=720,
                                            buttons=["fullscreen"])
                with gr.Accordion("다른 사진 불러와서 색감만 보정하기", open=False):
                    grade_upload = gr.Image(label="사진 올리기", type="numpy", image_mode="RGB",
                                            sources=["upload", "clipboard"], height=220)
            with gr.Column(scale=2):
                preset = gr.Dropdown(list(G.PRESETS), value="원본 그대로", label="프리셋",
                                     info="고르면 아래 슬라이더가 맞춰집니다. 고른 뒤 세부 조절도 가능해요.")
                auto_wb = gr.Checkbox(label="자동 색 균형 (누렇거나 푸르게 뜬 색 바로잡기)", value=False)
                with gr.Accordion("빛", open=True):
                    exposure = gr.Slider(-100, 100, 0, step=1, label="노출 (전체 밝기)")
                    contrast = gr.Slider(-100, 100, 0, step=1, label="대비")
                    highlights = gr.Slider(-100, 100, 0, step=1, label="밝은 영역",
                                           info="낮추면 하늘·흰옷의 날아간 부분이 살아납니다.")
                    shadows = gr.Slider(-100, 100, 0, step=1, label="어두운 영역",
                                        info="높이면 그늘진 얼굴이 밝아집니다.")
                with gr.Accordion("색", open=True):
                    temperature = gr.Slider(-100, 100, 0, step=1, label="색온도 (← 차갑게 · 따뜻하게 →)")
                    tint = gr.Slider(-100, 100, 0, step=1, label="색조 (← 초록 · 자주 →)")
                    saturation = gr.Slider(-100, 100, 0, step=1, label="채도", info="-100이면 흑백이 됩니다.")
                    vibrance = gr.Slider(-100, 100, 0, step=1, label="생동감",
                                         info="색이 옅은 곳만 살려서 피부색이 덜 과해집니다.")
                with gr.Accordion("효과", open=False):
                    sepia = gr.Slider(0, 100, 0, step=1, label="세피아 톤 (옛날 갈색 사진 느낌)")
                    fade = gr.Slider(0, 100, 0, step=1, label="바랜 느낌 (필름 톤)")
                    vignette = gr.Slider(0, 100, 0, step=1, label="비네팅 (가장자리 어둡게)")
                    sharpness = gr.Slider(0, 100, 0, step=1, label="선명도")
                with gr.Row():
                    btn_grade_reset = gr.Button("초기화", variant="secondary")
                    btn_grade_save = gr.Button("색감 적용해서 저장", variant="primary")
                grade_file = gr.File(label="색감 보정한 사진 내려받기 (PNG)")
                grade_info = gr.Markdown()
                grade_path = gr.State()
                del_g_btn, del_g_row, del_g_yes = delete_controls("저장한 색감 사진 삭제")

        # Grade 필드 순서와 같아야 함
        grade_ctrls = [auto_wb, exposure, temperature, tint, highlights, shadows, contrast,
                       saturation, vibrance, sepia, fade, vignette, sharpness]
        assert len(grade_ctrls) == len(G.FIELDS)
        preview_args = dict(fn=grade_preview, inputs=[grade_small] + grade_ctrls, outputs=grade_view,
                            show_progress="hidden")

        btn_mask.click(preview_mask, [editor, opts[1]], [mask_view, mask_info])
        btn_run.click(restore_one, [editor] + opts,
                      [slider, file_out, info, grade_src, grade_small, restore_path, del_r_btn]) \
            .then(**preview_args)
        del_r_yes.click(delete_restore, restore_path,
                        [slider, file_out, info, grade_src, grade_small, grade_view, restore_path,
                         del_r_btn, del_r_row])
        del_g_yes.click(delete_grade, grade_path, [grade_file, grade_info, grade_path, del_g_btn, del_g_row])
        for c in grade_ctrls:
            c.input(trigger_mode="always_last", **preview_args)
        preset.input(G.preset_values, preset, grade_ctrls).then(**preview_args)
        btn_grade_reset.click(lambda: ["원본 그대로"] + G.preset_values("원본 그대로"), None,
                              [preset] + grade_ctrls).then(**preview_args)
        grade_upload.upload(grade_load, grade_upload, [grade_src, grade_small]).then(**preview_args)
        btn_grade_save.click(grade_save, [grade_src] + grade_ctrls, [grade_file, grade_info, grade_path, del_g_btn])

    with gr.Tab("여러 장 한꺼번에"):
        gr.Markdown("여러 사진을 같은 설정으로 한 번에 복원합니다. 결과는 `결과` 폴더와 ZIP 파일로 저장됩니다.")
        with gr.Row():
            with gr.Column(scale=3):
                files = gr.File(label="사진 파일들", file_count="multiple", file_types=["image"])
                btn_batch = gr.Button("모두 복원", variant="primary", size="lg")
                gallery = gr.Gallery(label="결과", columns=4, height=480)
            with gr.Column(scale=2):
                bopts = option_panel(batch=True)
                with gr.Accordion("6. 색감 보정", open=False):
                    bpreset = gr.Dropdown(list(G.PRESETS), value="원본 그대로", label="모든 사진에 적용할 색감 프리셋")
                zip_out = gr.File(label="전체 결과 ZIP")
                blog = gr.Textbox(label="처리 기록", lines=8)
                batch_folder = gr.State()
                del_b_btn, del_b_row, del_b_yes = delete_controls("이번 결과 전체 삭제 (폴더 + ZIP)")
        btn_batch.click(restore_batch, [files, bpreset] + bopts, [gallery, zip_out, blog, batch_folder, del_b_btn])
        del_b_yes.click(delete_batch, batch_folder, [gallery, zip_out, blog, batch_folder, del_b_btn, del_b_row])

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
