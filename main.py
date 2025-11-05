import sys
import os
import numpy as np

# 使用绝对路径添加src目录
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(current_dir, 'src')
sys.path.insert(0, src_dir)

print(f"当前目录: {current_dir}")
print(f"添加路径: {src_dir}")

try:
    from feature_extractor import VideoFeatureExtractor  # 注意这里改了！
    print("✅ 成功导入特征提取器")
except ImportError as e:
    print(f"❌ 导入失败: {e}")
    print("请检查 feature_extractor.py 文件是否存在")
    exit(1)

def main():
    print("=== 基于内容的视频检索系统 ===")
    
    # 初始化特征提取器
    extractor = VideoFeatureExtractor()  # 注意这里改了！
    
    # 设置UCF101数据路径
    ucf101_path = r"E:\Users\Lenovo\Downloads\UCF101"
    
    # 检查路径是否存在
    if not os.path.exists(ucf101_path):
        print(f"❌ 路径不存在: {ucf101_path}")
        print("请确认UCF101数据集路径是否正确")
        return
    
    # 获取所有视频文件
    video_files = []
    for root, dirs, files in os.walk(ucf101_path):
        for file in files:
            if file.endswith(('.avi', '.mp4', '.mkv')):
                video_files.append(os.path.join(root, file))
    
    print(f"找到 {len(video_files)} 个视频文件")
    
    # 为了快速测试，先只处理前3个视频
    test_videos = video_files[:3]
    print("测试视频:", [os.path.basename(v) for v in test_videos])
    
    # 提取视频特征
    video_features = {}
    for video_path in test_videos:
        feature = extractor.process_video(video_path)
        if feature is not None:
            video_name = os.path.basename(video_path)
            video_features[video_name] = feature
            print(f"✅ 已处理: {video_name}")
    
    print(f"成功处理 {len(video_features)} 个视频")
    
    # 保存特征到文件
    if video_features:
        # 确保outputs文件夹存在
        os.makedirs("outputs", exist_ok=True)
        
        # 保存特征字典
        np.save("outputs/video_features.npy", video_features)
        print("✅ 特征已保存到 outputs/video_features.npy")
        
        # 打印特征维度信息
        sample_feature = list(video_features.values())[0]
        print(f"特征维度: {sample_feature.shape}")

if __name__ == "__main__":
    main()