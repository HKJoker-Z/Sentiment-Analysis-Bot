import torch
import transformers

print(f"PyTorch 版本: {torch.__version__}")
print(f"Transformers 版本: {transformers.__version__}")

try:
    from transformers import pipeline
    nlp = pipeline("sentiment-analysis", device=-1)
    result = nlp("I love this!")
    print("测试成功! 结果:", result)
except Exception as e:
    print(f"测试失败: {e}")