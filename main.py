# main.py
import sys
import os
import numpy as np

# 使用绝对路径添加src目录
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(current_dir, 'src')
sys.path.insert(0, src_dir)

print(f"当前目录: {current_dir}")
print(f"源文件目录: {src_dir}")

from feature_extractor import VideoFeatureExtractor
from retrieval import VideoRetrievalSystem

def extract_features_pipeline():
    """特征提取流水线"""
    print("\n" + "="*50)
    print("特征提取流水线")
    print("="*50)
    
    # 初始化特征提取器（启用CLIP）
    extractor = VideoFeatureExtractor(use_clip=True)
    
    # 视频数据路径 - 请修改为您的实际路径
    video_dirs = [
        r"E:\Users\Lenovo\Downloads\ucf101",
        # 可以添加更多路径
    ]
    
    # 收集视频文件
    video_files = []
    for video_dir in video_dirs:
        if not os.path.exists(video_dir):
            print(f"⚠️ 路径不存在: {video_dir}")
            continue
            
        print(f"扫描目录: {video_dir}")
        for root, dirs, files in os.walk(video_dir):
            for file in files:
                if file.lower().endswith(('.avi', '.mp4', '.mov', '.mkv')):
                    # 可选：按类别过滤
                    if any(action in root.lower() for action in ['apply', 'archery', 'basketball', 'cycling']):
                        video_files.append(os.path.join(root, file))
                    
                    # 限制数量用于测试
                    if len(video_files) >= 20:  # 增加到20个视频
                        break
            if len(video_files) >= 20:
                break
        if len(video_files) >= 20:
            break
    
    print(f"找到 {len(video_files)} 个视频文件")
    
    if not video_files:
        print("❌ 未找到任何视频文件，请检查路径配置")
        return None
    
    # 批量提取特征
    print("\n开始特征提取...")
    features_dict = extractor.extract_features_batch(video_files)
    
    print(f"\n特征提取完成: {len(features_dict)}/{len(video_files)} 个视频成功")
    
    # 保存特征
    os.makedirs("outputs", exist_ok=True)
    output_path = "outputs/video_features.npy"
    np.save(output_path, features_dict)
    print(f"特征已保存到: {output_path}")
    
    return output_path

def retrieval_demo(features_path):
    """检索演示"""
    print("\n" + "="*50)
    print("视频检索演示")
    print("="*50)
    
    try:
        # 初始化检索系统
        retrieval_system = VideoRetrievalSystem(features_path, clip_weight=0.5)
        
        # 显示系统信息
        info = retrieval_system.get_feature_info()
        print(f"视频数量: {info['video_count']}")
        print(f"传统特征维度: {info['traditional_feature_dim']}")
        print(f"CLIP特征维度: {info['clip_feature_dim']}")
        print(f"CLIP特征可用: {info['has_clip_features']}")
        
        if info['video_count'] == 0:
            print("❌ 没有可用的视频特征")
            return
        
        # 交互式检索演示
        while True:
            print("\n" + "-"*40)
            print("检索选项:")
            print("1. 自动测试查询")
            print("2. 选择特定视频查询") 
            print("3. 比较权重配置")
            print("4. 退出")
            
            choice = input("请选择操作 (1-4): ").strip()
            
            if choice == '1':
                # 自动测试查询
                test_video = retrieval_system.video_names[0]
                print(f"\n自动查询: {test_video}")
                retrieval_system.query_by_example(test_video, top_k=5)
                
            elif choice == '2':
                # 选择特定视频查询
                print(f"\n可用视频 (共{len(retrieval_system.video_names)}个):")
                for i, name in enumerate(retrieval_system.video_names[:10]):  # 显示前10个
                    print(f"  {i+1}. {name}")
                if len(retrieval_system.video_names) > 10:
                    print("  ...")
                
                try:
                    vid_choice = int(input("选择视频编号: ")) - 1
                    if 0 <= vid_choice < len(retrieval_system.video_names):
                        selected_video = retrieval_system.video_names[vid_choice]
                        retrieval_system.query_by_example(selected_video, top_k=5)
                    else:
                        print("❌ 编号无效")
                except ValueError:
                    print("❌ 请输入有效数字")
                    
            elif choice == '3':
                # 权重配置比较
                if len(retrieval_system.video_names) > 0:
                    test_video = retrieval_system.video_names[0]
                    weight_configs = [
                        (0.9, 0.1, "强侧重传统特征"),
                        (0.7, 0.3, "侧重传统特征"),
                        (0.5, 0.5, "平衡权重"), 
                        (0.3, 0.7, "侧重CLIP语义"),
                        (0.1, 0.9, "强侧重CLIP语义")
                    ]
                    retrieval_system.compare_weight_configs(test_video, weight_configs)
                else:
                    print("❌ 没有可用的视频")
                    
            elif choice == '4':
                print("退出检索演示")
                break
            else:
                print("❌ 请选择 1-4")
                
    except Exception as e:
        print(f"❌ 检索演示失败: {e}")

def system_status():
    """检查系统状态"""
    print("\n" + "="*50)
    print("系统状态检查")
    print("="*50)
    
    # 检查特征文件
    features_path = "outputs/video_features.npy"
    if os.path.exists(features_path):
        try:
            features_dict = np.load(features_path, allow_pickle=True).item()
            video_count = len(features_dict)
            has_clip = any(feat.get('clip') is not None for feat in features_dict.values())
            
            print(f"✅ 特征文件存在: {features_path}")
            print(f"   视频数量: {video_count}")
            print(f"   CLIP特征: {'可用' if has_clip else '不可用'}")
            
            # 显示特征维度
            if video_count > 0:
                sample_feat = next(iter(features_dict.values()))
                if sample_feat.get('traditional') is not None:
                    print(f"   传统特征维度: {sample_feat['traditional'].shape}")
                if sample_feat.get('clip') is not None:
                    print(f"   CLIP特征维度: {sample_feat['clip'].shape}")
                    
            return features_path
            
        except Exception as e:
            print(f"❌ 特征文件损坏: {e}")
            return None
    else:
        print("❌ 特征文件不存在")
        return None

def main():
    """主函数"""
    print("="*60)
    print("          双语义视频检索系统")
    print("="*60)
    
    while True:
        print("\n主菜单:")
        print("1. 特征提取流水线")
        print("2. 视频检索演示") 
        print("3. 系统状态检查")
        print("4. 运行完整演示 (提取+检索)")
        print("5. 退出")
        
        choice = input("请选择操作 (1-5): ").strip()
        
        if choice == '1':
            # 特征提取
            features_path = extract_features_pipeline()
            if features_path:
                print(f"✅ 特征提取完成: {features_path}")
                
        elif choice == '2':
            # 检索演示
            features_path = "outputs/video_features.npy"
            if os.path.exists(features_path):
                retrieval_demo(features_path)
            else:
                print("❌ 特征文件不存在，请先运行特征提取")
                
        elif choice == '3':
            # 系统状态检查
            system_status()
            
        elif choice == '4':
            # 完整演示
            features_path = extract_features_pipeline()
            if features_path:
                retrieval_demo(features_path)
                
        elif choice == '5':
            print("感谢使用视频检索系统！")
            break
            
        else:
            print("❌ 请选择 1-5")

if __name__ == "__main__":
    main()