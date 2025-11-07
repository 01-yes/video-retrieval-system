# Video Retrieval System

A semantic-based video retrieval system implementing content-based similarity search on UCF101 dataset.

## 📋 Project Overview
- **Task**: Video similarity retrieval based on semantic representation
- **Dataset**: UCF101 Action Recognition Dataset
- **Features**: OpenCV global features (color histograms, HOG, statistical features)
- **Similarity**: Cosine similarity metric

## 🏗️ Project Structure
```
video-retrieval-system/
├── src/
│   ├── feature_extractor.py  # Video feature extraction
│   └── retrieval.py          # Similarity search system
├── main.py                   # Main processing pipeline
├── demo.py                   # Demonstration script
├── test_clip.py             # Environment testing
└── .gitignore               # Git ignore rules
```

## 🚀 Quick Start
```bash
# Extract features from videos
python main.py

# Run retrieval system
python src/retrieval.py

# Full demonstration
python demo.py
```

## 📊 Results
- Achieved >0.98 similarity for same-action videos
- Successfully processed UCF101 dataset
- Implemented complete video retrieval pipeline

## 👨‍💻 Author
- GOODLAB Laboratory Assessment
- GitHub: [01-yes](https://github.com/01-yes)
