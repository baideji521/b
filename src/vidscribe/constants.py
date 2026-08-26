"""不依赖第三方库的公共常量（GUI 进程不应该被迫导入 cv2/torch）。"""

VIDEO_SUFFIXES = {".mp4", ".mov", ".mkv", ".avi", ".m4v", ".webm", ".flv", ".wmv", ".ts"}

# Qwen3-VL: patch_size(16) * merge_size(2)，所有分辨率必须对齐到 32
PIXEL_FACTOR = 32
