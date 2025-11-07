import cv2
import numpy as np
import open_clip
import torch
from PIL import Image
import os

class CLIPFeatureExtractor:
    def __init__(self):
        print("正在初始化CLIP特征提取器...")
        self.model, _, self.preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='openai')
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"使用设备: {self.device}")
        self.model.to(self.device)
        self.model.eval()
    
    def extract_frame_features(self, frame):
        """从单帧提取CLIP特征"""
        # 转换BGR到RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image_pil = Image.fromarray(rgb_frame)
        
        # 预处理
        image_tensor = self.preprocess(image_pil).unsqueeze(0).to(self.device)
        
        # 提取特征
        with torch.no_grad():
            features = self.model.encode_image(image_tensor)
            features = features.cpu().numpy().flatten()
        
        # 归一化
        norm = np.linalg.norm(features)
        if norm > 0:
            features = features / norm
            
        return features
    
    def process_video(self, video_path, sample_rate=5):
        """处理整个视频提取CLIP特征"""
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
                    clip_feat = self.extract_frame_features(frame)
                    frame_features.append(clip_feat)
                except Exception as e:
                    print(f"处理帧时出错: {e}")
                    continue
            
            frame_count += 1
        
        cap.release()
        
        # 对帧特征进行平均池化
        if frame_features:
            video_feature = np.mean(frame_features, axis=0)
            # 再次归一化
            norm = np.linalg.norm(video_feature)
            if norm > 0:
                video_feature = video_feature / norm
            print(f"CLIP特征提取完成，维度: {video_feature.shape}")
            return video_feature
        else:
            print("无法从视频中提取CLIP特征")
            return None