try:
    import seaborn as sns
    print("成功导入 seaborn!")
except ImportError as e:
    print(f"导入 seaborn 失败: {e}")

# 尝试导入其他依赖项
try:
    import pandas as pd
    import matplotlib.pyplot as plt
    import streamlit as st
    import torch
    from transformers import pipeline
    print("所有其他依赖项导入成功!")
except ImportError as e:
    print(f"导入失败: {e}")