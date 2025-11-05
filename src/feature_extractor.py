import cv2
import numpy as np
import os

class VideoFeatureExtractor:
    def __init__(self):
        """初始化特征提取器"""
        print("正在初始化视频特征提取器...")
        self.feature_type = "opencv_global"
        print("✅ 特征提取器初始化完成（使用OpenCV全局特征）")
    
    def extract_frame_features(self, frame):
        """提取单帧的全局特征"""
        # 调整帧大小
        frame_resized = cv2.resize(frame, (224, 224))
        
        # 方法1: 颜色直方图特征
        hist_features = self._extract_color_histogram(frame_resized)
        
        # 方法2: HOG特征
        hog_features = self._extract_hog_features(frame_resized)
        
        # 方法3: 全局像素统计特征
        stat_features = self._extract_statistical_features(frame_resized)
        
        # 合并所有特征
        combined_features = np.concatenate([hist_features, hog_features, stat_features])
        
        return combined_features
    
    def _extract_color_histogram(self, frame):
        """提取颜色直方图特征"""
        # 转换为HSV颜色空间
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # 计算HSV三个通道的直方图
        hist_h = cv2.calcHist([hsv], [0], None, [16], [0, 180])
        hist_s = cv2.calcHist([hsv], [1], None, [16], [0, 256])
        hist_v = cv2.calcHist([hsv], [2], None, [16], [0, 256])
        
        # 归一化并展平
        hist_h = cv2.normalize(hist_h, hist_h).flatten()
        hist_s = cv2.normalize(hist_s, hist_s).flatten()
        hist_v = cv2.normalize(hist_v, hist_v).flatten()
        
        return np.concatenate([hist_h, hist_s, hist_v])
    
    def _extract_hog_features(self, frame):
        """提取HOG特征"""
        # 转换为灰度图
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # 计算HOG特征
        hog = cv2.HOGDescriptor((64, 64), (16, 16), (8, 8), (8, 8), 9)
        hog_features = hog.compute(gray)
        
        if hog_features is not None:
            return hog_features.flatten()
        else:
            return np.zeros(1764)  # HOG特征的标准维度
    
    def _extract_statistical_features(self, frame):
        """提取统计特征"""
        # 转换为HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        features = []
        for i in range(3):  # 对H,S,V三个通道
            channel = hsv[:, :, i]
            features.extend([
                np.mean(channel),    # 均值
                np.std(channel),     # 标准差
                np.median(channel),  # 中位数
                np.min(channel),     # 最小值
                np.max(channel)      # 最大值
            ])
        
        return np.array(features)
    
    def process_video(self, video_path, sample_rate=5):
        """处理整个视频，提取全局特征"""
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
            print(f"✅ 成功提取特征，维度: {video_feature.shape}")
            return video_feature
        else:
            print(f"⚠️ 无法从视频中提取特征: {os.path.basename(video_path)}")
            return None

if __name__ == "__main__":
    # 测试代码
    extractor = VideoFeatureExtractor()
    print("✅ 特征提取器初始化成功！")