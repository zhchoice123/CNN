#!/usr/bin/env bash
# 启动训练实验室 + 学习站（同一个端口）。浏览器打开 http://localhost:8766/#/lab
cd "$(dirname "$0")/.."
exec python3 lab/server.py
