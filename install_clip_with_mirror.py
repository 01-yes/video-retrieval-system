# install_clip_with_mirror.py
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

print("🔧 使用国内镜像安装CLIP...")

try:
    import open_clip
    import torch
    print("✅ CLIP已安装，正在测试...")
    
    # 设置环境变量使用镜像
    os.environ['TRANSFORMERS_OFFLINE'] = '0'
    os.environ['HF_HUB_ENABLE_PROGRESS_BARS'] = '1'
    
    model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='openai')
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    print(f"🎉 CLIP加载成功！使用设备: {device}")
    
except Exception as e:
    print(f"❌ 失败: {e}")