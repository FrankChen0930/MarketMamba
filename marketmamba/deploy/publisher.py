"""
MarketMamba V5.5 — GitHub 自動推送模組
負責：克隆 Repo → 複製更新檔案 → Commit → Push → 觸發 Streamlit 更新
"""

import os
import logging

from marketmamba.config import (
    GITHUB_USER, GITHUB_EMAIL, GITHUB_REPO_URL_TEMPLATE,
    GITHUB_PUSH_FILES, get_now_str, get_repo_output_dir, is_colab,
)

logger = logging.getLogger('MarketMamba.publisher')


def _ensure_update_time(output_dir: str) -> None:
    """確保 update_time.txt 存在 (推送前自動生成)"""
    path = os.path.join(output_dir, 'update_time.txt')
    tw_time = get_now_str()
    with open(path, 'w') as f:
        f.write(tw_time)
    logger.info(f"🕒 update_time.txt 已生成: {tw_time}")


def push_to_github(github_token: str = None) -> bool:
    """
    自動推送更新至 GitHub

    流程：
    1. 確保 update_time.txt 已生成
    2. 克隆 Repo 到暫存目錄
    3. 從輸出目錄複製更新的檔案 (跳過不存在的)
    4. git add → commit → push

    Args:
        github_token: GitHub Personal Access Token
                      如未傳入，嘗試從 Colab userdata 取得

    Returns:
        是否推送成功
    """
    print("🚀 開始執行自動化發布管線...")

    # 取得 Token
    if github_token is None:
        if is_colab():
            try:
                from google.colab import userdata
                github_token = userdata.get('GITHUB_TOKEN')
            except Exception as e:
                logger.error(f"❌ 無法取得 GITHUB_TOKEN: {e}")
                return False
        else:
            github_token = os.environ.get('GITHUB_TOKEN', '')

    if not github_token:
        logger.error("❌ 缺少 GITHUB_TOKEN，無法推送")
        return False

    repo_url = GITHUB_REPO_URL_TEMPLATE.format(
        token=github_token, user=GITHUB_USER
    )

    # 0. 先確保 update_time.txt 存在
    output_dir = get_repo_output_dir()
    _ensure_update_time(output_dir)

    # 1. 克隆
    work_dir = '/content/repo_publish' if is_colab() else os.path.join(os.getcwd(), '_repo_publish')
    os.system(f"rm -rf {work_dir}")
    clone_result = os.system(f"git clone {repo_url} {work_dir}")
    if clone_result != 0:
        logger.error("❌ git clone 失敗")
        return False

    # 2. 複製更新的檔案 (跳過不存在的，不再報 warning)
    copied = []
    skipped = []
    for filename in GITHUB_PUSH_FILES:
        src = os.path.join(output_dir, filename)
        if os.path.exists(src):
            os.system(f'cp "{src}" "{work_dir}/"')
            copied.append(filename)
        else:
            skipped.append(filename)

    if skipped:
        logger.info(f"⏭️ 跳過未生成的檔案: {', '.join(skipped)}")
    logger.info(f"📦 已複製 {len(copied)} 個檔案")

    # 3. 確保 work_dir 也有 update_time.txt
    tw_time = get_now_str()
    with open(os.path.join(work_dir, "update_time.txt"), "w") as f:
        f.write(tw_time)

    # 4. Git 操作
    original_dir = os.getcwd()
    os.chdir(work_dir)

    os.system(f"git config user.name '{GITHUB_USER}'")
    os.system(f"git config user.email '{GITHUB_EMAIL}'")

    # 只 add 實際存在的檔案
    files_to_add = copied + ['update_time.txt']
    for f in files_to_add:
        os.system(f"git add {f}")

    # Commit + Push
    os.system(f"git commit -m '🤖 Auto-update: V5.5 Pipeline ({tw_time})'")
    push_result = os.system("git push origin main")

    os.chdir(original_dir)

    if push_result == 0:
        print(f"🎉 大功告成！網頁與資料庫已同步至最新狀態！({tw_time})")
        return True
    else:
        logger.error("⚠️ git push 失敗！請檢查 Token 權限")
        return False


def shutdown_colab() -> None:
    """Colab 專用：切斷執行階段電源，釋放 GPU"""
    if is_colab():
        import time
        print("🔌 系統將在 3 秒後自動切斷執行階段電源...")
        time.sleep(3)
        from google.colab import runtime
        runtime.unassign()
