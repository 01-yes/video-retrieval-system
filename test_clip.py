import open_clip
import torch

print("=== CLIP 测试 ===")

try:
    print("1. 导入open_clip...")
    print("✅ open_clip 导入成功")
    print("版本:", open_clip.__version__)

    print("2. 加载CLIP模型...")
    model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='openai')
    print("✅ CLIP模型加载成功")
    
    print("3. 检查设备...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("设备:", device)
    
    print("4. 测试图像编码器...")
    # 创建一个随机图像进行测试
    import numpy as np
    from PIL import Image
    random_image = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
    image_tensor = preprocess(random_image).unsqueeze(0)
    
    with torch.no_grad():
        features = model.encode_image(image_tensor)
        print("✅ 图像编码器工作正常")
        print("特征维度:", features.shape)
    
    print("🎉 所有测试通过！CLIP完全可用")

except Exception as e:
    print(f"❌ 测试失败: {e}")
    print("错误类型:", type(e).__name__)