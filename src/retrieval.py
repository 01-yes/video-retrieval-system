import numpy as np
import os

class VideoRetrievalSystem:
    def __init__(self, features_path):
        """初始化检索系统"""
        print("正在加载视频特征...")
        self.video_features = np.load(features_path, allow_pickle=True).item()
        self.video_names = list(self.video_features.keys())
        self.feature_vectors = np.array(list(self.video_features.values()))
        print(f"✅ 加载了 {len(self.video_names)} 个视频的特征")
        print(f"特征维度: {self.feature_vectors.shape}")
    
    def cosine_similarity(self, vec1, vec2):
        """计算余弦相似度"""
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        return dot_product / (norm1 * norm2)
    
    def retrieve_similar_videos(self, query_feature, top_k=3):
        """检索相似视频"""
        similarities = []
        
        for i, video_feature in enumerate(self.feature_vectors):
            similarity = self.cosine_similarity(query_feature, video_feature)
            similarities.append((self.video_names[i], similarity))
        
        # 按相似度排序
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        return similarities[:top_k]
    
    def query_by_example(self, example_video_name, top_k=3):
        """以视频为例进行查询"""
        if example_video_name not in self.video_features:
            print(f"❌ 未找到视频: {example_video_name}")
            return None
        
        query_feature = self.video_features[example_video_name]
        print(f"查询视频: {example_video_name}")
        
        results = self.retrieve_similar_videos(query_feature, top_k)
        
        print("\n=== 检索结果 ===")
        for i, (video_name, similarity) in enumerate(results):
            print(f"{i+1}. {video_name} (相似度: {similarity:.4f})")
        
        return results

def main():
    # 测试检索系统
    features_path = "outputs/video_features.npy"
    
    if not os.path.exists(features_path):
        print("❌ 特征文件不存在，请先运行 main.py 提取特征")
        return
    
    retrieval_system = VideoRetrievalSystem(features_path)
    
    # 测试检索
    if retrieval_system.video_names:
        print("\n=== 测试检索功能 ===")
        # 用第一个视频作为查询示例
        query_video = retrieval_system.video_names[0]
        retrieval_system.query_by_example(query_video)

if __name__ == "__main__":
    main()