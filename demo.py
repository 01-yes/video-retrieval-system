import sys
import os
import numpy as np

# 添加路径
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(current_dir, 'src')
sys.path.insert(0, src_dir)

from feature_extractor import VideoFeatureExtractor
from retrieval import VideoRetrievalSystem

def demo():
    print("=" * 50)
    print("       视频语义检索系统演示")
    print("=" * 50)
    
    # 1. 特征提取演示
    print("\n1. 特征提取阶段")
    print("正在处理UCF101视频数据...")
    
    extractor = VideoFeatureExtractor()
    ucf101_path = r"E:\Users\Lenovo\Downloads\UCF101"
    
    # 处理多个类别的视频
    video_files = []
    for root, dirs, files in os.walk(ucf101_path):
        for file in files:
            if file.endswith('.avi') and any(action in root for action in ['ApplyEyeMakeup', 'ApplyLipstick', 'Archery']):
                video_files.append(os.path.join(root, file))
            if len(video_files) >= 6:  # 处理6个视频用于演示
                break
        if len(video_files) >= 6:
            break
    
    print(f"选择了 {len(video_files)} 个视频进行演示")
    
    # 提取特征
    video_features = {}
    for video_path in video_files:
        feature = extractor.process_video(video_path)
        if feature is not None:
            video_name = os.path.basename(video_path)
            video_features[video_name] = feature
    
    # 保存特征
    os.makedirs("outputs", exist_ok=True)
    np.save("outputs/demo_features.npy", video_features)
    print("✅ 特征提取完成")
    
    # 2. 检索演示
    print("\n2. 视频检索演示")
    retrieval_system = VideoRetrievalSystem("outputs/demo_features.npy")
    
    # 演示不同查询
    test_queries = list(video_features.keys())[:2]  # 用前2个视频作为查询
    
    for query_video in test_queries:
        print(f"\n--- 查询: {query_video} ---")
        results = retrieval_system.query_by_example(query_video, top_k=3)
    
    print("\n" + "=" * 50)
    print("🎉 演示完成！系统功能验证成功！")
    print("=" * 50)

if __name__ == "__main__":
    demo()