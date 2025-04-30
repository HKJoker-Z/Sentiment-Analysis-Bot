#!/usr/bin/env python
import os
import sys
import subprocess

# 指定虚拟环境中 Python 解释器的路径
VENV_PYTHON = "/Users/z2/Other/Python/pushing/venv/bin/python"  # 替换为你的实际路径

def main():
    # 获取项目根目录
    project_root = os.path.dirname(os.path.abspath(__file__))
    app_path = os.path.join(project_root, "src", "app.py")
    
    # 使用虚拟环境的 Python 运行 Streamlit
    subprocess.run([VENV_PYTHON, "-m", "streamlit", "run", app_path])

if __name__ == "__main__":
    main()