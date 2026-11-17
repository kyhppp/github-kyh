"""허깅페이스 스페이스에 옛날 사진 복원기를 올린다 (허깅페이스에_올리기.bat 이 실행).

처음엔 스페이스를 만들고, 그다음부터는 바뀐 파일만 다시 올린다.
"""
from __future__ import annotations

import sys
from pathlib import Path

from huggingface_hub import HfApi
from huggingface_hub.errors import LocalTokenNotFoundError

import gradio

ROOT = Path(__file__).resolve().parent
SPACE_NAME = "photo-restorer"

# 스페이스 설정은 README 맨 위 YAML 머리말로 정한다 (GitHub 용 README는 그대로 둠)
HEADER = f"""---
title: Old Photo Restorer
emoji: 🖼️
colorFrom: yellow
colorTo: gray
sdk: gradio
sdk_version: {gradio.__version__}
python_version: "3.12"
app_file: app.py
pinned: false
---

"""

IGNORE = [
    ".venv/**", "models/**", "결과/**", ".gradio/**", ".git/**",
    "**/__pycache__/**", "*.pyc", "*.part",
    "비밀번호.txt", "*.bat", "hf_deploy.py", "README.md",
]


def main() -> int:
    api = HfApi()
    try:
        user = api.whoami()["name"]
    except LocalTokenNotFoundError:
        print("허깅페이스에 로그인되어 있지 않아요.")
        return 1
    repo_id = f"{user}/{SPACE_NAME}"

    print(f"스페이스 준비: {repo_id}")
    # 혼자 쓰는 용도라 비공개: 허깅페이스에 로그인한 본인만 열 수 있다
    api.create_repo(repo_id, repo_type="space", space_sdk="gradio", private=True, exist_ok=True)
    api.update_repo_settings(repo_id, repo_type="space", private=True)

    readme = HEADER + (ROOT / "README.md").read_text(encoding="utf-8")
    api.upload_file(path_or_fileobj=readme.encode("utf-8"), path_in_repo="README.md",
                    repo_id=repo_id, repo_type="space", commit_message="README")
    print("파일 올리는 중...")
    api.upload_folder(folder_path=ROOT, repo_id=repo_id, repo_type="space",
                      ignore_patterns=IGNORE, commit_message="앱 업데이트")

    url = f"https://huggingface.co/spaces/{repo_id}"
    print("\n================================================================")
    print(f"  고정 주소:  {url}")
    print("  (서버가 준비되는 데 처음엔 5~10분 걸려요. 'Building' 이 끝나면 열려요)")
    print("  비공개 주소라 허깅페이스에 로그인한 상태에서만 열려요.")
    print("================================================================\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
