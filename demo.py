# demo.py
import sys
import os
import numpy as np

# 添加路径，确保可以导入src中的模块
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(current_dir, 'src')
sys.path.insert(0, src_dir)

from feature_extractor import VideoFeatureExtractor
from retrieval import VideoRetrievalSystem

def demo():
    print("=" * 60)
    print("          双语义视频检索系统演示")
    print("=" * 60)

    # 1. 特征提取演示
    print("\n1. 特征提取阶段")
    print("正在处理视频数据...")

    # 初始化特征提取器（启用CLIP）
    extractor = VideoFeatureExtractor(use_clip=True)
    
    ucf101_path = r"E:\Users\Lenovo\Downloads\ucf101"
    
    # 收集视频文件 - 这里以6个视频为例
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
        features = extractor.extract_features(video_path)
        if features is not None:
            video_name = os.path.basename(video_path)
            video_features[video_name] = features
            print(f"✓ 已处理: {video_name}")

    # 保存特征
    os.makedirs("outputs", exist_ok=True)
    np.save("outputs/demo_features.npy", video_features)
    print("✓ 特征提取完成，已保存到 outputs/demo_features.npy")

    # 2. 检索演示
    print("\n2. 视频检索演示")
    
    try:
        retrieval_system = VideoRetrievalSystem("outputs/demo_features.npy", clip_weight=0.5)
        
        # 演示不同权重配置的查询
        test_queries = list(video_features.keys())[:2]  # 用前2个视频测试
        
        for query_video in test_queries:
            print(f"\n--- 查询视频: {query_video} ---")
            
            # 测试不同权重配置
            weight_configs = [
                (0.8, 0.2, "侧重传统特征"),
                (0.5, 0.5, "平衡权重"), 
                (0.2, 0.8, "侧重CLIP语义")
            ]
            
            for trad_w, clip_w, desc in weight_configs:
                retrieval_system.set_weights(trad_w, clip_w)
                results = retrieval_system.retrieve_similar_videos(
                    video_features[query_video], top_k=3
                )
                
                print(f"\n【{desc}】")
                for i, (name, combined_sim, trad_sim, clip_sim) in enumerate(results):
                    if retrieval_system.has_clip_features:
                        print(f"  {i+1}. {name}")
                        print(f"     综合相似度: {combined_sim:.4f} (传统: {trad_sim:.4f}, CLIP: {clip_sim:.4f})")
                    else:
                        print(f"  {i+1}. {name} (相似度: {combined_sim:.4f})")

        print("\n" + "=" * 60)
        print("✓ 演示完成! 系统功能验证成功！")
        print("=" * 60)
        
    except Exception as e:
        print(f"✗ 检索演示失败: {e}")
        print("请确保特征文件已正确生成")

def simple_demo():
    """简化版演示，适用于快速测试"""
    print("=" * 50)
    print("    简化版视频检索演示")
    print("=" * 50)
    
    # 检查特征文件是否存在
    features_path = "outputs/demo_features.npy"
    if not os.path.exists(features_path):
        print("特征文件不存在，请先运行完整演示或提取特征")
        return
    
    # 初始化检索系统
    retrieval_system = VideoRetrievalSystem(features_path, clip_weight=0.5)
    
    # 显示系统信息
    info = retrieval_system.get_feature_info()
    print(f"已加载 {info['video_count']} 个视频的特征")
    print(f"CLIP特征可用: {info['has_clip_features']}")
    
    # 进行查询
    if retrieval_system.video_names:
        query_video = retrieval_system.video_names[0]
        print(f"\n正在查询: {query_video}")
        
        results = retrieval_system.query_by_example(query_video, top_k=3)
        
        return results
    else:
        print("没有可用的视频进行查询")
        return None

if __name__ == "__main__":
    # 运行完整演示
    demo()
    # 或运行简化演示
    # simple_demo()