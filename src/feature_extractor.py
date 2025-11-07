import cv2
import numpy as np
import os
import sys

# 添加当前路径以便导入其他模块
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

try:
    from clip_extractor import CLIPFeatureExtractor
    CLIP_AVAILABLE = True
except ImportError:
    print("⚠️ CLIP模块不可用，将使用传统特征")
    CLIP_AVAILABLE = False

class VideoFeatureExtractor:
    def __init__(self, use_clip=True):
        """初始化特征提取器"""
        print("正在初始化视频特征提取器...")
        
        self.use_clip = use_clip and CLIP_AVAILABLE
        self.clip_extractor = None  # 添加这行
        
        if self.use_clip:
            try:
                from clip_extractor import CLIPFeatureExtractor
                self.clip_extractor = CLIPFeatureExtractor()  # 创建CLIP提取器实例
                self.feature_type = "opencv_global + CLIP"
                print("✅ CLIP特征提取器加载成功")
            except Exception as e:
                print(f"❌ CLIP加载失败: {e}")
                self.use_clip = False
                self.feature_type = "opencv_global"
        else:
            self.feature_type = "opencv_global"
            print("使用传统特征提取器")
        
        print(f"特征类型: {self.feature_type}")
        
        print(f"特征类型: {self.feature_type}")

    def extract_frame_features(self, frame):
        """提取单帧的全局特征（传统方法）"""
        # 调整大小
        frame_resized = cv2.resize(frame, (224, 224))
        
        # 方法1: 颜色直方图特征
        hist_features = self._extract_color_histogram(frame_resized)
        
        # 方法2: HOG特征
        hog_features = self._extract_hog_features(frame_resized)
        
        # 方法3: 全局像素统计特征
        stat_features = self._extract_statistical_features(frame_resized)
        
        # 合并所有传统特征
        combined_features = np.concatenate([hist_features, hog_features, stat_features])
        
        return combined_features

    def _extract_color_histogram(self, frame):
        """提取颜色直方图特征"""
        # 转换为HSV颜色空间
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # 计算HSV三个通道的直方图
        hist_h = cv2.calcHist([hsv], [0], None, [50], [0, 180])
        hist_s = cv2.calcHist([hsv], [1], None, [50], [0, 256])
        hist_v = cv2.calcHist([hsv], [2], None, [50], [0, 256])
        
        # 归一化并展平
        hist_h = cv2.normalize(hist_h, hist_h).flatten()
        hist_s = cv2.normalize(hist_s, hist_s).flatten()
        hist_v = cv2.normalize(hist_v, hist_v).flatten()
        
        # 合并三个通道的直方图
        color_features = np.concatenate([hist_h, hist_s, hist_v])
        return color_features

    def _extract_hog_features(self, frame):
        """提取HOG特征"""
        # 转换为灰度图
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # 初始化HOG描述符
        win_size = (64, 64)
        block_size = (16, 16)
        block_stride = (8, 8)
        cell_size = (8, 8)
        nbins = 9
        
        hog = cv2.HOGDescriptor(win_size, block_size, block_stride, cell_size, nbins)
        
        # 计算HOG特征
        hog_features = hog.compute(gray)
        
        if hog_features is not None:
            return hog_features.flatten()
        else:
            # 返回零向量作为备用
            return np.zeros(1764)  # HOG特征的标准维度

    def _extract_statistical_features(self, frame):
        """提取统计特征"""
        # 转换为HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        features = []
        for i in range(3):  # HSV三个通道
            channel = hsv[:, :, i]
            features.extend([
                np.mean(channel),    # 均值
                np.std(channel),     # 标准差
                np.median(channel),  # 中位数
                np.min(channel),     # 最小值
                np.max(channel)      # 最大值
            ])
        
        return np.array(features)

    def process_video(self, video_path, sample_rate=3):
        """处理单个视频，提取全局特征（传统方法）"""
        print(f"处理视频: {os.path.basename(video_path)}")
        
        cap = cv2.VideoCapture(video_path)
        frame_features = []
        frame_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # 按采样率抽取帧
            if frame_count % sample_rate == 0:
                try:
                    feature = self.extract_frame_features(frame)
                    frame_features.append(feature)
                except Exception as e:
                    print(f"处理帧时出错: {e}")
                    continue
            
            frame_count += 1
        
        cap.release()
        
        # 对所有帧特征进行平均池化
        if frame_features:
            video_feature = np.mean(frame_features, axis=0)
            # 归一化
            norm = np.linalg.norm(video_feature)
            if norm > 0:
                video_feature = video_feature / norm
            print(f"✓ 传统特征提取完成，维度: {video_feature.shape}")
            return video_feature
        else:
            print(f"✗ 无法从视频中提取特征: {os.path.basename(video_path)}")
            return None

    def extract_features(self, video_path):
        """提取视频的多种特征（传统 + CLIP）"""
        print(f"\n开始提取特征: {os.path.basename(video_path)}")
        
        # 传统特征
        traditional_feat = self.process_video(video_path)
        
        # CLIP特征
        clip_feat = None
        if self.use_clip:
            try:
                clip_feat = self.clip_extractor.process_video(video_path)
                if clip_feat is not None:
                    print(f"✓ CLIP特征提取完成，维度: {clip_feat.shape}")
                else:
                    print("⚠️ CLIP特征提取失败")
            except Exception as e:
                print(f"✗ CLIP特征提取出错: {e}")
                clip_feat = None
        
        result = {
            'traditional': traditional_feat,
            'clip': clip_feat,
            'video_name': os.path.basename(video_path)
        }
        
        # 检查特征是否有效
        if traditional_feat is None:
            print(f"✗ 特征提取完全失败: {os.path.basename(video_path)}")
            return None
        
        return result

    def extract_features_batch(self, video_paths):
        """批量提取特征"""
        features_dict = {}
        
        for video_path in video_paths:
            features = self.extract_features(video_path)
            if features is not None:
                features_dict[features['video_name']] = features
        
        return features_dict


# 测试代码
if __name__ == "__main__":
    # 测试特征提取器
    extractor = VideoFeatureExtractor(use_clip=True)
    print("特征提取器初始化成功！")
    
    # 测试单帧特征提取
    test_frame = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    frame_features = extractor.extract_frame_features(test_frame)
    print(f"单帧特征维度: {frame_features.shape}")
    
    # 测试传统特征组件
    color_feat = extractor._extract_color_histogram(test_frame)
    hog_feat = extractor._extract_hog_features(test_frame)
    stat_feat = extractor._extract_statistical_features(test_frame)
    
    print(f"颜色特征维度: {color_feat.shape}")
    print(f"HOG特征维度: {hog_feat.shape}")
    print(f"统计特征维度: {stat_feat.shape}")
    print(f"总特征维度: {frame_features.shape}")