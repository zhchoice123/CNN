#!/usr/bin/env bash
# macOS / Linux：一次性准备训练环境（逻辑都在 setup.py 里，Windows 用 setup_windows.bat）
exec python3 "$(dirname "$0")/setup.py" "$@"
