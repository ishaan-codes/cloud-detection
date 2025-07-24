# Cloud Mask Prediction from EO Images using CloudSEN 12 Dataset

## Team Members
- Ishaan Gupta (ishaangupta.246@gmail.com)

## Project Overview
This project develops a deep learning solution for detecting and segmenting cloud-covered areas in Sentinel-2 satellite imagery. The system classifies pixels into four categories: clear sky, thick clouds, thin clouds, and cloud shadows. The implementation features robust data handling with TacoReader API, a lightweight U-Net model, and comprehensive error recovery mechanisms for reliable training in resource-constrained environments.

## Instructions to Run Code

### 1. Prerequisites
- Python 3.8+
- NVIDIA GPU (recommended) or CPU

### 2. Installation
```bash
git clone https://github.com/yourusername/cloud-mask-prediction.git
cd cloud-mask-prediction
pip install -r requirements.txt
```

### 3. Training the Model
```bash
python train_cloud_mask_model.py
```

### 4. Key Arguments
```bash
python train_cloud_mask_model.py \
  --batch_size 4 \
  --epochs 30 \
  --output_dir results
```

## Implementation Details

### i. Data Preprocessing Steps

#### On-Demand Streaming:
- Accesses Sentinel-2 imagery directly from cloud storage via TacoReader API
- No local dataset download required

#### Band Selection:
- Uses 6 spectral bands: B2 (Blue), B3 (Green), B4 (Red), B8 (NIR), B11, B12 (SWIR)

#### Image Processing:
- Normalization: Pixel values scaled by 10,000
- Resizing: Bilinear interpolation to 256x256 pixels
- Error Handling: Automatic retries with exponential backoff

#### Robustness Features:
- Rate limit detection with 5-minute cooldowns
- Placeholder samples for failed loads
- Synthetic data fallback mode

### ii. Models Used
- Lightweight U-Net Architecture:
- Encoder: 3 downsampling blocks (32→64→128→256 channels)
- Decoder: 3 upsampling blocks with skip connections

## Key Features:
- 4-class output (clear/thick cloud/thin cloud/shadow)
- Class-weighted loss function
- Mixed precision training
- Parameters: ~1.2 million
- Performance: Processes 4 samples/batch on standard GPU

## Tools and Frameworks
- Deep Learning: PyTorch 2.1.0
- Geospatial Processing: Rasterio, TacoReader
- Image Processing: Pillow, OpenCV
- Visualization: Matplotlib
- Metrics: scikit-learn
- Utilities: NumPy, tqdm
