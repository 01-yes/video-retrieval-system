# core_experiments.py
"""
🎯 视频检索系统 - 核心实验验证
4个必要验证：环境依赖、特征提取、检索功能、权重效果
"""

import os
import sys
import time
import numpy as np
import cv2

# 设置环境变量
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

# 添加路径
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(current_dir, 'src')
sys.path.insert(0, src_dir)

from feature_extractor import VideoFeatureExtractor
from retrieval import VideoRetrievalSystem

class CoreExperiments:
    """核心实验验证"""
    
    def __init__(self):
        self.results = {}
        print("🎬 视频检索系统核心实验验证")
        print("=" * 50)
    
    def experiment_1_environment(self):
        """实验1: 环境依赖验证"""
        print("\n1. 🔧 环境依赖验证")
        print("-" * 30)
        
        deps = {
            'opencv-python': 'cv2',
            'numpy': 'numpy', 
            'torch': 'torch',
            'open_clip': 'open_clip'
        }
        
        all_ok = True
        for pkg, name in deps.items():
            try:
                __import__(name)
                print(f"   ✅ {pkg}")
            except:
                print(f"   ❌ {pkg}")
                all_ok = False
        
        self.results['environment'] = all_ok
        return all_ok
    
    def experiment_2_feature_extraction(self):
        """实验2: 特征提取验证"""
        print("\n2. 🔍 特征提取验证")
        print("-" * 30)
        
        try:
            # 测试双特征提取器
            extractor = VideoFeatureExtractor(use_clip=True)
            print(f"   ✅ 提取器初始化 - {extractor.feature_type}")
            
            # 测试特征提取
            test_frame = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
            
            # 传统特征
            trad_feat = extractor.extract_frame_features(test_frame)
            print(f"   ✅ 传统特征 - 维度: {trad_feat.shape}")
            
            # CLIP特征
            if extractor.use_clip:
                clip_feat = extractor.clip_extractor.extract_frame_features(test_frame)
                print(f"   ✅ CLIP特征 - 维度: {clip_feat.shape}")
            else:
                print("   ⚠️  CLIP特征 - 不可用")
            
            self.results['feature_extraction'] = True
            return True
            
        except Exception as e:
            print(f"   ❌ 特征提取失败: {e}")
            self.results['feature_extraction'] = False
            return False
    
    def experiment_3_retrieval_function(self):
        """实验3: 检索功能验证"""
        print("\n3. 📊 检索功能验证")
        print("-" * 30)
        
        try:
            # 创建测试数据
            test_data = {}
            video_names = ['运动视频1.mp4', '运动视频2.mp4', '其他视频1.mp4', '其他视频2.mp4']
            
            for i, name in enumerate(video_names):
                # 让前两个视频更相似
                if i < 2:
                    trad_feat = np.random.rand(1929) * 0.7 + 0.3
                    clip_feat = np.random.rand(512) * 0.8 + 0.2
                else:
                    trad_feat = np.random.rand(1929) * 0.3 + 0.1
                    clip_feat = np.random.rand(512) * 0.2 + 0.1
                
                # 归一化
                trad_feat = trad_feat / np.linalg.norm(trad_feat)
                clip_feat = clip_feat / np.linalg.norm(clip_feat)
                
                test_data[name] = {
                    'traditional': trad_feat.astype(np.float32),
                    'clip': clip_feat.astype(np.float32),
                    'video_name': name
                }
            
            # 保存测试数据
            os.makedirs("outputs", exist_ok=True)
            np.save("outputs/test_retrieval.npy", test_data)
            
            # 测试检索系统
            retrieval = VideoRetrievalSystem("outputs/test_retrieval.npy", clip_weight=0.5)
            print(f"   ✅ 检索系统初始化 - {retrieval.get_video_count()}个视频")
            
            # 执行检索
            query_video = video_names[0]
            results = retrieval.query_by_example(query_video, top_k=3)
            
            print(f"   🔍 查询: {query_video}")
            for i, (name, combined, trad, clip) in enumerate(results):
                print(f"      {i+1}. {name} - 相似度: {combined:.3f}")
            
            self.results['retrieval'] = True
            return True
            
        except Exception as e:
            print(f"   ❌ 检索功能失败: {e}")
            self.results['retrieval'] = False
            return False
    
    def experiment_4_weight_comparison(self):
        """实验4: 权重效果验证"""
        print("\n4. ⚖️ 权重效果验证")
        print("-" * 30)
        
        try:
            test_data = np.load("outputs/test_retrieval.npy", allow_pickle=True).item()
            retrieval = VideoRetrievalSystem("outputs/test_retrieval.npy")
            
            query_video = list(test_data.keys())[0]
            
            # 测试三种权重配置
            configs = [
                (0.8, 0.2, "传统侧重"),
                (0.5, 0.5, "平衡模式"), 
                (0.2, 0.8, "CLIP侧重")
            ]
            
            print(f"   查询视频: {query_video}")
            
            for trad_w, clip_w, desc in configs:
                retrieval.set_weights(trad_w, clip_w)
                results = retrieval.retrieve_similar_videos(test_data[query_video], top_k=2)
                
                if results:
                    best_match = results[0]
                    print(f"   【{desc}】最佳: {best_match[0]} - 相似度: {best_match[1]:.3f}")
            
            self.results['weight_comparison'] = True
            return True
            
        except Exception as e:
            print(f"   ❌ 权重验证失败: {e}")
            self.results['weight_comparison'] = False
            return False
    
    def run_all_experiments(self):
        """运行所有实验"""
        start_time = time.time()
        
        print("开始核心实验验证...")
        
        # 运行4个核心实验
        exp1 = self.experiment_1_environment()
        exp2 = self.experiment_2_feature_extraction()
        exp3 = self.experiment_3_retrieval_function() 
        exp4 = self.experiment_4_weight_comparison()
        
        # 汇总结果
        total_time = time.time() - start_time
        
        print("\n" + "=" * 50)
        print("📈 实验验证汇总")
        print("=" * 50)
        
        success_count = sum([exp1, exp2, exp3, exp4])
        print(f"✅ 通过实验: {success_count}/4")
        print(f"⏱️  总耗时: {total_time:.1f}秒")
        
        if success_count == 4:
            print("\n🎉 所有核心实验验证通过！")
            print("   系统功能完整，可以正常使用。")
        else:
            print(f"\n⚠️  {4-success_count}个实验未通过")
            print("   请检查相关功能模块。")
        
        return success_count == 4

def main():
    """主函数"""
    validator = CoreExperiments()
    success = validator.run_all_experiments()
    
    # 返回退出代码
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()