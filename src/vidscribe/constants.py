"""不依赖第三方库的公共常量（GUI 进程不应该被迫导入 cv2/torch）。"""

# 全项目唯一一份视频后缀名单：GUI 扫目录、缓存清单、数据库入库都认这一份，
# 免得某个后缀在一处能排队、另一处不登记。
VIDEO_SUFFIXES = {".mp4", ".mov", ".mkv", ".avi", ".m4v", ".webm", ".flv", ".wmv", ".ts",
                  ".mpg", ".mpeg"}

# Qwen3-VL: patch_size(16) * merge_size(2)，所有分辨率必须对齐到 32
PIXEL_FACTOR = 32
