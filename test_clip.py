# test_clip.py
import open_clip
import torch
import numpy as np
from PIL import Image
import sys
import os

def test_clip_functionality():
    """测试CLIP模型功能"""
    print("=" * 50)
    print("        CLIP 功能测试")
    print("=" * 50)
    
    try:
        # 1. 导入测试
        print("1. 导入open_clip...")
        print("✓ open_clip 导入成功")
        print("版本:", open_clip.__version__)
        
        # 2. 加载模型
        print("\n2. 加载CLIP模型...")
        model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='openai')
        print("✓ CLIP模型加载成功")
        
        # 3. 检查设备
        print("\n3. 检查设备...")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print("设备:", device)
        model.to(device)
        
        # 4. 测试图像编码器
        print("\n4. 测试图像编码器...")
        # 创建一个随机图像进行测试
        random_image = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
        image_tensor = preprocess(random_image).unsqueeze(0).to(device)
        
        with torch.no_grad():
            image_features = model.encode_image(image_tensor)
            print("✓ 图像编码器工作正常")
            print("特征维度:", image_features.shape)
        
        # 5. 测试文本编码器
        print("\n5. 测试文本编码器...")
        text_tokens = open_clip.tokenize(["a photo of a cat", "a picture of a dog"]).to(device)
        
        with torch.no_grad():
            text_features = model.encode_text(text_tokens)
            print("✓ 文本编码器工作正常")
            print("文本特征维度:", text_features.shape)
        
        # 6. 测试相似度计算
        print("\n6. 测试相似度计算...")
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        
        similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
        print("✓ 相似度计算正常")
        print("图像-文本相似度:", similarity.cpu().numpy())
        
        # 7. 性能测试
        print("\n7. 性能测试...")
        import time
        
        # 测试处理速度
        start_time = time.time()
        test_images = 10
        
        for i in range(test_images):
            test_img = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
            img_tensor = preprocess(test_img).unsqueeze(0).to(device)
            with torch.no_grad():
                _ = model.encode_image(img_tensor)
        
        end_time = time.time()
        avg_time = (end_time - start_time) / test_images
        print(f"平均处理时间: {avg_time:.3f} 秒/图像")
        print(f"预估FPS: {1/avg_time:.1f}")
        
        print("\n" + "=" * 50)
        print("🎉 所有测试通过！CLIP完全可用")
        print("=" * 50)
        
        return True
        
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        print("错误类型:", type(e).__name__)
        
        # 提供具体的错误解决方案
        if "CUDA" in str(e):
            print("\n💡 解决方案: 尝试使用CPU模式或检查CUDA安装")
        elif "download" in str(e).lower():
            print("\n💡 解决方案: 检查网络连接，或手动下载模型")
        elif "module" in str(e).lower():
            print("\n💡 解决方案: 重新安装依赖: pip install open-clip-torch")
        
        return False

def test_clip_with_real_image(image_path=None):
    """使用真实图像测试CLIP"""
    print("\n" + "=" * 50)
    print("     真实图像CLIP测试")
    print("=" * 50)
    
    try:
        model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='openai')
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)
        
        if image_path and os.path.exists(image_path):
            # 使用提供的图像
            image = Image.open(image_path).convert('RGB')
            print(f"测试图像: {os.path.basename(image_path)}")
        else:
            # 创建测试图像
            print("使用生成的测试图像")
            image = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
        
        # 预处理图像
        image_tensor = preprocess(image).unsqueeze(0).to(device)
        
        # 提取特征
        with torch.no_grad():
            features = model.encode_image(image_tensor)
            features = features.cpu().numpy().flatten()
        
        print(f"✓ 特征提取成功")
        print(f"特征向量维度: {features.shape}")
        print(f"特征范围: [{features.min():.3f}, {features.max():.3f}]")
        print(f"特征范数: {np.linalg.norm(features):.3f}")
        
        # 测试文本匹配
        texts = [
            "a photo of an animal",
            "a picture of a landscape", 
            "an image of a person",
            "a graphic design",
            "a random pattern"
        ]
        
        text_tokens = open_clip.tokenize(texts).to(device)
        
        with torch.no_grad():
            text_features = model.encode_text(text_tokens)
            # 归一化
            image_features_norm = features / np.linalg.norm(features)
            text_features_norm = text_features / text_features.norm(dim=-1, keepdim=True)
            
            # 计算相似度
            similarities = (image_features_norm @ text_features_norm.cpu().numpy().T).flatten()
        
        print(f"\n📊 文本匹配结果:")
        for i, (text, sim) in enumerate(zip(texts, similarities)):
            print(f"  {i+1}. '{text}' -> 相似度: {sim:.4f}")
        
        return True
        
    except Exception as e:
        print(f"❌ 真实图像测试失败: {e}")
        return False

def check_dependencies():
    """检查依赖包"""
    print("=" * 50)
    print("       依赖包检查")
    print("=" * 50)
    
    dependencies = {
        'torch': '深度学习框架',
        'open_clip': 'CLIP模型',
        'PIL': '图像处理',
        'numpy': '数值计算'
    }
    
    all_ok = True
    for package, description in dependencies.items():
        try:
            if package == 'PIL':
                __import__('PIL.Image')
            else:
                __import__(package)
            print(f"✓ {package:15} - {description}")
        except ImportError:
            print(f"❌ {package:15} - 未安装")
            all_ok = False
    
    return all_ok

def main():
    """主测试函数"""
    print("🎬 CLIP功能完整性测试")
    print("此脚本将测试CLIP模型的所有核心功能")
    print()
    
    # 检查依赖
    if not check_dependencies():
        print("\n❌ 缺少依赖包，请先安装:")
        print("pip install open-clip-torch torch torchvision pillow numpy")
        return
    
    print("\n开始CLIP功能测试...")
    
    # 运行基本功能测试
    basic_ok = test_clip_functionality()
    
    # 运行真实图像测试
    real_image_ok = test_clip_with_real_image()
    
    print("\n" + "=" * 60)
    print("             测试总结")
    print("=" * 60)
    
    if basic_ok and real_image_ok:
        print("🎉 所有测试通过！CLIP可以正常使用")
        print("\n下一步:")
        print("1. 运行 demo.py 进行系统演示")
        print("2. 运行 main.py 使用完整系统")
    else:
        print("⚠️  部分测试失败，请检查上述错误信息")
        print("\n常见解决方案:")
        print("1. 检查网络连接")
        print("2. 重新安装依赖: pip install --upgrade open-clip-torch")
        print("3. 检查CUDA安装（如使用GPU）")

if __name__ == "__main__":
    main()