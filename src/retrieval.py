import numpy as np
import os

class VideoRetrievalSystem:
    def __init__(self, features_path, clip_weight=0.5):
        """初始化检索系统"""
        print("正在加载视频特征...")
        
        if not os.path.exists(features_path):
            print(f"✗ 特征文件不存在: {features_path}")
            raise FileNotFoundError(f"特征文件 {features_path} 不存在")
        
        try:
            features_dict = np.load(features_path, allow_pickle=True).item()
        except Exception as e:
            print(f"✗ 加载特征文件失败: {e}")
            raise
        
        self.video_names = []
        self.traditional_features = []
        self.clip_features = []
        self.has_clip_features = False
        
        # 分析特征结构
        successful_loads = 0
        for video_name, feature_dict in features_dict.items():
            if feature_dict is None:
                continue
                
            traditional_feat = feature_dict.get('traditional')
            clip_feat = feature_dict.get('clip')
            
            if traditional_feat is not None:
                self.video_names.append(video_name)
                self.traditional_features.append(traditional_feat)
                
                # 处理CLIP特征
                if clip_feat is not None and np.any(clip_feat):
                    self.clip_features.append(clip_feat)
                    self.has_clip_features = True
                else:
                    # 如果没有CLIP特征，创建零向量
                    if self.has_clip_features and len(self.clip_features) > 0:
                        # 如果之前有视频有CLIP特征，这个视频没有，创建相同维度的零向量
                        clip_dim = self.clip_features[0].shape[0]
                        self.clip_features.append(np.zeros(clip_dim))
                    else:
                        # 第一个视频或所有视频都没有CLIP特征
                        self.clip_features.append(np.zeros(512))  # 默认CLIP维度
                
                successful_loads += 1
        
        if successful_loads == 0:
            print("✗ 没有成功加载任何视频特征")
            raise ValueError("没有有效的视频特征")
        
        # 转换为numpy数组
        self.traditional_features = np.array(self.traditional_features)
        self.clip_features = np.array(self.clip_features)
        
        # 设置权重
        self.set_weights(1.0 - clip_weight, clip_weight)
        
        print(f"✓ 成功加载 {len(self.video_names)} 个视频的特征")
        print(f"传统特征维度: {self.traditional_features.shape}")
        print(f"CLIP特征维度: {self.clip_features.shape}")
        print(f"CLIP特征可用: {self.has_clip_features}")
        print(f"特征权重 - 传统: {self.traditional_weight:.2f}, CLIP: {self.clip_weight:.2f}")

    def set_weights(self, traditional_weight, clip_weight):
        """动态设置特征权重"""
        total_weight = traditional_weight + clip_weight
        if total_weight == 0:
            traditional_weight = clip_weight = 0.5
        else:
            traditional_weight /= total_weight
            clip_weight /= total_weight
        
        self.traditional_weight = traditional_weight
        self.clip_weight = clip_weight
        
        # 如果没有CLIP特征，强制使用传统特征
        if not self.has_clip_features and clip_weight > 0:
            print("⚠️ 没有CLIP特征，将只使用传统特征")
            self.traditional_weight = 1.0
            self.clip_weight = 0.0
        
        print(f"✓ 权重已更新 - 传统: {self.traditional_weight:.2f}, CLIP: {self.clip_weight:.2f}")

    def cosine_similarity(self, vec1, vec2):
        """计算余弦相似度"""
        if vec1 is None or vec2 is None:
            return 0.0
            
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 > 0 and norm2 > 0:
            similarity = dot_product / (norm1 * norm2)
            # 确保相似度在合理范围内
            return max(0.0, min(1.0, similarity))
        else:
            return 0.0

    def retrieve_similar_videos(self, query_features, top_k=5):
        """检索相似视频"""
        if query_features is None or query_features.get('traditional') is None:
            print("✗ 查询特征无效")
            return []
        
        query_traditional = query_features['traditional']
        query_clip = query_features.get('clip')
        
        # 如果查询没有CLIP特征但数据库有，创建零向量
        if query_clip is None and self.has_clip_features:
            query_clip = np.zeros(self.clip_features.shape[1])
        
        similarities = []
        
        for i in range(len(self.video_names)):
            # 计算传统特征相似度
            trad_sim = self.cosine_similarity(query_traditional, self.traditional_features[i])
            
            # 计算CLIP特征相似度
            clip_sim = 0.0
            if self.clip_weight > 0 and query_clip is not None:
                clip_sim = self.cosine_similarity(query_clip, self.clip_features[i])
            
            # 加权融合
            combined_sim = (self.traditional_weight * trad_sim + 
                          self.clip_weight * clip_sim)
            
            similarities.append((self.video_names[i], combined_sim, trad_sim, clip_sim))
        
        # 按综合相似度排序
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        # 返回前top_k个结果，但不超过总数
        return similarities[:min(top_k, len(similarities))]

    def query_by_example(self, example_video_name, top_k=5):
        """以视频为例进行查询"""
        if example_video_name not in self.video_names:
            print(f"✗ 未找到视频: {example_video_name}")
            print(f"可用视频: {self.video_names[:5]}...")  # 只显示前5个
            return None
        
        # 获取查询视频的索引
        query_idx = self.video_names.index(example_video_name)
        
        # 构建查询特征
        query_features = {
            'traditional': self.traditional_features[query_idx],
            'clip': self.clip_features[query_idx] if self.has_clip_features else None
        }
        
        print(f"🔍 查询视频: {example_video_name}")
        print(f"检索设置: top_k={top_k}, 传统权重={self.traditional_weight:.2f}, CLIP权重={self.clip_weight:.2f}")
        
        results = self.retrieve_similar_videos(query_features, top_k)
        
        self._display_results(results)
        return results

    def query_by_features(self, traditional_feature, clip_feature=None, top_k=5):
        """直接通过特征向量进行查询"""
        query_features = {
            'traditional': traditional_feature,
            'clip': clip_feature
        }
        
        print(f"🔍 特征向量查询")
        print(f"检索设置: top_k={top_k}, 传统权重={self.traditional_weight:.2f}, CLIP权重={self.clip_weight:.2f}")
        
        results = self.retrieve_similar_videos(query_features, top_k)
        
        self._display_results(results)
        return results

    def _display_results(self, results):
        """显示检索结果"""
        if not results:
            print("✗ 没有找到相似视频")
            return
        
        print("\n" + "="*60)
        print("📊 检索结果")
        print("="*60)
        
        for i, (video_name, combined_sim, trad_sim, clip_sim) in enumerate(results):
            print(f"{i+1:2d}. {video_name}")
            print(f"     综合相似度: {combined_sim:.4f}", end="")
            if self.has_clip_features:
                print(f" (传统: {trad_sim:.4f}, CLIP: {clip_sim:.4f})")
            else:
                print(f" (传统: {trad_sim:.4f})")
        
        print("="*60)

    def get_video_count(self):
        """获取视频数量"""
        return len(self.video_names)

    def get_feature_info(self):
        """获取特征信息"""
        info = {
            'video_count': len(self.video_names),
            'traditional_feature_dim': self.traditional_features.shape[1] if len(self.traditional_features) > 0 else 0,
            'clip_feature_dim': self.clip_features.shape[1] if len(self.clip_features) > 0 else 0,
            'has_clip_features': self.has_clip_features,
            'weights': {
                'traditional': self.traditional_weight,
                'clip': self.clip_weight
            }
        }
        return info

    def compare_weight_configs(self, query_video_name, weight_configs):
        """比较不同权重配置的检索结果"""
        if query_video_name not in self.video_names:
            print(f"✗ 未找到视频: {query_video_name}")
            return
        
        original_trad_weight = self.traditional_weight
        original_clip_weight = self.clip_weight
        
        print(f"\n🎯 权重配置比较 - 查询视频: {query_video_name}")
        print("="*70)
        
        for trad_w, clip_w, desc in weight_configs:
            self.set_weights(trad_w, clip_w)
            results = self.query_by_example(query_video_name, top_k=3)
            
            if results:
                best_match = results[0]  # 最相似的结果
                print(f"【{desc}】最佳匹配: {best_match[0]} (相似度: {best_match[1]:.4f})")
        
        # 恢复原始权重
        self.set_weights(original_trad_weight, original_clip_weight)
        print("="*70)


# 测试代码
if __name__ == "__main__":
    # 测试检索系统
    try:
        features_path = "outputs/dual_features.npy"
        retrieval_system = VideoRetrievalSystem(features_path, clip_weight=0.5)
        
        print("\n" + "="*50)
        print("系统信息:")
        info = retrieval_system.get_feature_info()
        for key, value in info.items():
            print(f"  {key}: {value}")
        
        # 如果有视频，测试查询
        if retrieval_system.get_video_count() > 0:
            print("\n" + "="*50)
            print("测试检索功能:")
            
            # 用第一个视频测试
            test_video = retrieval_system.video_names[0]
            retrieval_system.query_by_example(test_video, top_k=3)
            
            # 测试不同权重
            print("\n" + "="*50)
            print("权重配置比较:")
            weight_configs = [
                (0.8, 0.2, "侧重传统特征"),
                (0.5, 0.5, "平衡权重"), 
                (0.2, 0.8, "侧重CLIP语义")
            ]
            retrieval_system.compare_weight_configs(test_video, weight_configs)
            
        print("\n✓ 检索系统测试完成")
        
    except Exception as e:
        print(f"✗ 检索系统测试失败: {e}")