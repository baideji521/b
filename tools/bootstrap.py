"""无人值守安装 / 校验 / 冒烟测试。全部逻辑放在 Python 里，.bat 只做最薄的引导。

用法（由 setup_and_test.bat / run_auto.bat 调用，也可手工执行）：
    python tools/bootstrap.py --all
    python tools/bootstrap.py --install --verify
    python tools/bootstrap.py --smoke
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
LOG_DIR = ROOT / "logs"
sys.path.insert(0, str(SRC))

TORCH_SPEC = ["torch==2.8.0", "torchvision==0.23.0", "torchaudio==2.8.0"]
MAX_RETRIES = 3

# 默认值；实际取 config.json 的 mirrors 段（优先国内镜像）
DEFAULT_MIRRORS = {
    "pypi": [
        "https://pypi.tuna.tsinghua.edu.cn/simple",
        "https://mirrors.aliyun.com/pypi/simple",
        "https://pypi.org/simple",
    ],
    "pytorch_index": [
        "https://mirror.nju.edu.cn/pytorch/whl/cu126",
        "https://download.pytorch.org/whl/cu126",
    ],
    "pytorch_find_links": ["https://mirrors.aliyun.com/pytorch-wheels/cu126/"],
}


def load_mirrors() -> dict:
    mirrors = dict(DEFAULT_MIRRORS)
    cfg_file = ROOT / "config.json"
    if cfg_file.is_file():
        try:
            with open(cfg_file, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            for key, value in (data.get("mirrors") or {}).items():
                if value:
                    mirrors[key] = value
        except Exception:
            pass
    return mirrors


MIRRORS = load_mirrors()

_log_file: Path | None = None


def log(message: str) -> None:
    line = f"{datetime.now():%H:%M:%S} {message}"
    print(line, flush=True)
    if _log_file is not None:
        with open(_log_file, "a", encoding="utf-8") as fh:
            fh.write(line + "\n")


def init_log(name: str) -> None:
    global _log_file
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    _log_file = LOG_DIR / f"{name}_{datetime.now():%Y%m%d_%H%M%S}.log"


def run(cmd: list[str], timeout: int = 7200) -> int:
    log("$ " + " ".join(cmd))
    try:
        proc = subprocess.run(cmd, cwd=str(ROOT), timeout=timeout, check=False,
                              capture_output=True, text=True, encoding="utf-8", errors="replace")
    except subprocess.TimeoutExpired:
        log(f"[FAIL] 命令超时({timeout}s)")
        return 124
    tail = (proc.stdout or "")[-3000:] + (proc.stderr or "")[-3000:]
    if _log_file is not None:
        with open(_log_file, "a", encoding="utf-8") as fh:
            fh.write((proc.stdout or "") + (proc.stderr or "") + "\n")
    if proc.returncode != 0:
        log(f"[FAIL] 退出码 {proc.returncode}\n{tail}")
    return proc.returncode


def pip(*args: str, index_url: str | None = None, extra_index: str | None = None,
        find_links: str | None = None) -> int:
    cmd = [sys.executable, "-m", "pip", "install", "--disable-pip-version-check", *args]
    if index_url:
        cmd += ["--index-url", index_url]
    if extra_index:
        cmd += ["--extra-index-url", extra_index]
    if find_links:
        cmd += ["-f", find_links]
    return run(cmd)


def _pypi() -> list[str]:
    return list(MIRRORS.get("pypi") or DEFAULT_MIRRORS["pypi"])


# ------------------------------------------------------------------ 步骤
def step_env() -> bool:
    log("=" * 60)
    log("步骤 1/6：环境检查")
    log(f"OS      : {platform.system()} {platform.release()}")
    log(f"Python  : {sys.version.split()[0]}  ({sys.executable})")
    if sys.version_info < (3, 10):
        log("[FAIL] 需要 Python >= 3.10")
        return False
    try:
        proc = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version",
                               "--format=csv,noheader"], capture_output=True, text=True, timeout=30, check=False)
        if proc.returncode == 0:
            log(f"GPU     : {proc.stdout.strip()}")
        else:
            log("[WARN] nvidia-smi 不可用，可能没有 NVIDIA 驱动，将回退 CPU")
    except Exception:
        log("[WARN] 未检测到 nvidia-smi，将回退 CPU")
    return True


def step_install() -> bool:
    log("=" * 60)
    log("步骤 2/6：安装依赖（优先国内镜像）")
    pypi = _pypi()
    log("PyPI 镜像顺序: " + " -> ".join(pypi))
    pip("-U", "pip", "setuptools", "wheel", index_url=pypi[0], extra_index=pypi[-1])

    try:
        import torch  # noqa: PLC0415

        log(f"torch 已安装: {torch.__version__} (cuda={torch.version.cuda}, available={torch.cuda.is_available()})")
    except ImportError:
        if not _install_torch():
            return False

    for attempt in range(1, MAX_RETRIES + 1):
        mirror = pypi[min(attempt - 1, len(pypi) - 1)]
        log(f"安装 requirements.txt（index={mirror}），第 {attempt}/{MAX_RETRIES} 次")
        if pip("-r", str(ROOT / "requirements.txt"), index_url=mirror, extra_index=pypi[-1]) == 0:
            return True
    log("[FAIL] 依赖安装失败")
    return False


def _install_torch() -> bool:
    """PyTorch (CUDA 12.6)：国内镜像优先，官方源兜底。"""
    pypi = _pypi()
    indexes = list(MIRRORS.get("pytorch_index") or DEFAULT_MIRRORS["pytorch_index"])
    find_links = list(MIRRORS.get("pytorch_find_links") or [])

    for index in indexes:
        log(f"安装 PyTorch，index={index}")
        if pip(*TORCH_SPEC, index_url=index) == 0:
            return True
        log(f"[WARN] {index} 安装失败，换下一个源")
    for link in find_links:
        # 阿里云等镜像是文件列表页，只能用 -f，index 仍走国内 PyPI
        log(f"安装 PyTorch，find-links={link}")
        if pip(*TORCH_SPEC, index_url=pypi[0], extra_index=pypi[-1], find_links=link) == 0:
            return True
        log(f"[WARN] {link} 安装失败")
    log("[FAIL] PyTorch 安装失败（所有镜像与官方源都试过了）")
    return False


def step_verify() -> bool:
    log("=" * 60)
    log("步骤 3/6：校验依赖与 API")
    code = (
        "import json, torch, transformers, cv2, numpy, PIL\n"
        "import qwen_vl_utils, faster_whisper, ctranslate2\n"
        "from qwen_vl_utils import process_vision_info\n"
        "import inspect\n"
        "sig = list(inspect.signature(process_vision_info).parameters)\n"
        "info = {\n"
        "  'torch': torch.__version__, 'cuda': torch.version.cuda,\n"
        "  'cuda_available': torch.cuda.is_available(),\n"
        "  'transformers': transformers.__version__,\n"
        "  'has_qwen3vl': hasattr(transformers, 'Qwen3VLForConditionalGeneration'),\n"
        "  'process_vision_info_params': sig,\n"
        "  'faster_whisper': faster_whisper.__version__ if hasattr(faster_whisper,'__version__') else 'n/a',\n"
        "  'ctranslate2': ctranslate2.__version__,\n"
        "  'ct2_cuda_types': sorted(ctranslate2.get_supported_compute_types('cuda')) if torch.cuda.is_available() else [],\n"
        "}\n"
        "print('VERIFY_JSON=' + json.dumps(info))\n"
        "assert info['has_qwen3vl'], 'transformers 缺少 Qwen3VLForConditionalGeneration，需要 >= 4.57.0'\n"
    )
    proc = subprocess.run([sys.executable, "-c", code], cwd=str(ROOT), capture_output=True,
                          text=True, encoding="utf-8", errors="replace", check=False)
    output = (proc.stdout or "") + (proc.stderr or "")
    if _log_file is not None:
        with open(_log_file, "a", encoding="utf-8") as fh:
            fh.write(output + "\n")
    for line in output.splitlines():
        if line.startswith("VERIFY_JSON="):
            log("依赖版本: " + line[len("VERIFY_JSON="):])
    if proc.returncode != 0:
        log(f"[FAIL] 依赖校验失败\n{output[-2000:]}")
        return False
    log("[OK] 依赖校验通过")
    return True


def step_download(full: bool) -> bool:
    log("=" * 60)
    log("步骤 4/6：下载并验证模型（ModelScope -> hf-mirror -> 官方）")
    args = [sys.executable, "-m", "vidscribe.cli", "download"]
    if full:
        args.append("--all")
    env_pythonpath = os.environ.get("PYTHONPATH", "")
    os.environ["PYTHONPATH"] = str(SRC) + (os.pathsep + env_pythonpath if env_pythonpath else "")
    pypi = _pypi()
    os.environ.setdefault("PIP_INDEX_URL", pypi[0])
    for attempt in range(1, MAX_RETRIES + 1):
        if run(args) == 0:
            log("[OK] 模型下载完成")
            return True
        log(f"[WARN] 模型下载失败，重试 {attempt}/{MAX_RETRIES}")
        time.sleep(3)
    log("[FAIL] 模型下载失败")
    return False


def step_smoke() -> bool:
    """图片 / 视频帧 / Whisper 三个最小验证，不依赖真实素材。"""
    log("=" * 60)
    log("步骤 5/6：冒烟测试（图片 + 视频采样 + Whisper）")
    os.environ.setdefault("PYTHONPATH", str(SRC))
    code = r"""
import sys, json
sys.path.insert(0, r"{src}")
import numpy as np
from PIL import Image
from vidscribe.config import Config
from vidscribe.logging_setup import setup_logging
from pathlib import Path

cfg = Config.load(r"{root}")
cfg.ensure_dirs()
setup_logging(cfg.path("log_dir"), name="smoke")
report = {{}}

# --- 1. 视频探测 + 内存帧采样（不落地 JPG）---
from vidscribe.video_io import list_videos, probe_video, plan_frame_indices, sample_frames
videos = list_videos(cfg.path("input_dir")) or sorted(Path(r"{root}").glob("*.mp4"))
if videos:
    info = probe_video(videos[0])
    idx = plan_frame_indices(info, 0.0, min(info.duration, 8.0), 2.0, 4, 8)
    batch = sample_frames(info, idx, 200*32*32)
    report["video_probe"] = {{"file": info.name, "duration": info.duration, "fps": info.fps,
                             "frames_sampled": len(batch), "timestamps": batch.timestamps[:4],
                             "has_audio": info.has_audio}}
    assert len(batch) > 0, "内存帧采样失败"
    test_image = batch.images[0]
else:
    report["video_probe"] = "no video found"
    test_image = Image.fromarray((np.random.rand(256, 256, 3) * 255).astype("uint8"))

# --- 2. 视觉模型：单图描述 ---
from vidscribe.visual.qwen_vl import QwenVLAnalyzer
analyzer = QwenVLAnalyzer(cfg.visual, str(cfg.path("model_dir")), cfg.mirrors)
analyzer.load()
messages = [{{"role": "user", "content": [{{"type": "image", "image": test_image}},
                                          {{"type": "text", "text": "用一句话描述这张图片。"}}]}}]
text = analyzer.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = analyzer.processor(text=[text], images=[test_image], return_tensors="pt").to(analyzer.model.device)
import torch
with torch.inference_mode():
    out = analyzer.model.generate(**inputs, max_new_tokens=64, do_sample=False)
answer = analyzer.processor.batch_decode([out[0][len(inputs["input_ids"][0]):]], skip_special_tokens=True)[0]
report["image_test"] = {{"model": analyzer.model_id, "answer": answer.strip()[:200]}}
assert answer.strip(), "视觉模型没有输出"

# --- 3. Whisper 加载 ---
from vidscribe.speech.whisper_asr import WhisperASR
asr = WhisperASR(cfg.speech, str(cfg.path("model_dir")), cfg.mirrors)
asr.load()
report["whisper_test"] = {{"size": asr.model_size, "device": asr.device, "compute_type": asr.compute_type}}

print("SMOKE_JSON=" + json.dumps(report, ensure_ascii=False))
""".format(src=str(SRC), root=str(ROOT))

    proc = subprocess.run([sys.executable, "-c", code], cwd=str(ROOT), capture_output=True,
                          text=True, encoding="utf-8", errors="replace", check=False, timeout=3600)
    output = (proc.stdout or "") + (proc.stderr or "")
    if _log_file is not None:
        with open(_log_file, "a", encoding="utf-8") as fh:
            fh.write(output + "\n")
    for line in output.splitlines():
        if line.startswith("SMOKE_JSON="):
            log("冒烟结果: " + line[len("SMOKE_JSON="):])
    if proc.returncode != 0:
        log(f"[FAIL] 冒烟测试失败\n{output[-3000:]}")
        return False
    log("[OK] 冒烟测试通过")
    return True


def step_pipeline() -> bool:
    log("=" * 60)
    log("步骤 6/6：端到端流水线（Timeline + JSON/TXT/SRT + Benchmark）")
    env_pythonpath = os.environ.get("PYTHONPATH", "")
    if str(SRC) not in env_pythonpath:
        os.environ["PYTHONPATH"] = str(SRC) + (os.pathsep + env_pythonpath if env_pythonpath else "")
    rc = run([sys.executable, "-m", "vidscribe.cli", "run"])
    report = ROOT / "FINAL_REPORT.txt"
    if rc == 0 and report.is_file():
        log("[OK] 流水线完成，报告: " + str(report))
        return True
    log("[FAIL] 流水线未全部成功，详见 FINAL_REPORT.txt 与 logs/")
    return False


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--all", action="store_true", help="执行全部步骤")
    parser.add_argument("--install", action="store_true")
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--pipeline", action="store_true")
    parser.add_argument("--full-models", action="store_true", help="同时下载所有降级备用模型")
    parser.add_argument("--log-name", default="bootstrap")
    args = parser.parse_args()

    init_log(args.log_name)
    steps = []
    if args.all:
        steps = ["env", "install", "verify", "download", "smoke", "pipeline"]
    else:
        if args.install:
            steps += ["env", "install"]
        if args.verify:
            steps.append("verify")
        if args.download:
            steps.append("download")
        if args.smoke:
            steps.append("smoke")
        if args.pipeline:
            steps.append("pipeline")
    if not steps:
        steps = ["env", "verify"]

    results: dict[str, bool] = {}
    for step in steps:
        if step == "env":
            ok = step_env()
        elif step == "install":
            ok = step_install()
        elif step == "verify":
            ok = step_verify()
        elif step == "download":
            ok = step_download(args.full_models)
        elif step == "smoke":
            ok = step_smoke()
        else:
            ok = step_pipeline()
        results[step] = ok
        if not ok and step in ("env", "install", "verify"):
            log(f"[FAIL] 关键步骤 {step} 失败，终止")
            break

    log("=" * 60)
    log("汇总: " + json.dumps(results, ensure_ascii=False))
    failed = [k for k, v in results.items() if not v]
    if failed:
        log(f"[FAIL] 失败步骤: {failed}")
        return 1
    log("[OK] 全部步骤通过")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
