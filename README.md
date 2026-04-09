# Deepfake Detection via Remote Photoplethysmography (rPPG)

A research-grade pipeline for detecting deepfake videos by analyzing physiological signal consistency extracted from facial regions using remote photoplethysmography (rPPG).

## Motivation

Deepfake videos often fail to preserve consistent physiological signals (like heart rate patterns) across time and spatial regions of the face. This project exploits that weakness by:

1. Extracting rPPG signals from multiple facial regions (forehead, left cheek, right cheek)
2. Analyzing temporal and spatial consistency of these signals
3. Training an LSTM classifier to distinguish real from fake videos

## Pipeline Overview

```
Video Input
    |
    v
[1. Video Loader] --> Frame extraction at target FPS (30 Hz)
    |
    v
[2. Face Detection] --> OpenCV Haar cascade, stable bounding box
    |
    v
[3. ROI Extraction] --> Forehead (R1), Left Cheek (R2), Right Cheek (R3)
    |
    v
[4. rPPG Extraction] --> GREEN, CHROM, POS methods (3 ROIs x 3 methods = 9 signals)
    |
    v
[5. Signal Processing] --> Resample -> Detrend -> Normalize -> Bandpass (0.7-3.0 Hz)
    |
    v
[6. Segmentation] --> 30-second windows, optional SNR-based filtering
    |
    v
[7. EDA] --> Signal visualization, FFT analysis, quality metrics, ROI consistency
    |
    v
[8. Feature Engineering] --> Multivariate sequences (window_size x 9 features)
    |
    v
[9. LSTM Model] --> Bidirectional LSTM, binary classification
    |
    v
[10. Evaluation] --> Accuracy, ROC-AUC, 5-fold CV, confusion matrix
```

## Project Structure

```
deepfakeippg/
├── data/
│   ├── raw/                  # FaceForensics++ videos (not included)
│   ├── processed/            # Extracted features (generated)
│   └── cache/                # Cached intermediate results
├── notebooks/
│   └── eda_rppg_analysis.ipynb  # Exploratory Data Analysis
├── src/
│   ├── preprocessing/
│   │   ├── video_loader.py   # Video reading and frame sampling
│   │   ├── face_detector.py  # Face detection with OpenCV
│   │   └── roi_extractor.py  # ROI extraction (forehead, cheeks)
│   ├── rppg/
│   │   ├── green.py          # GREEN channel method
│   │   ├── chrom.py          # CHROM method (De Haan & Jeanne, 2013)
│   │   ├── pos.py            # POS method (Wang et al., 2017)
│   │   └── signal_processor.py  # Signal processing pipeline
│   ├── features/
│   │   ├── segmenter.py      # Signal segmentation
│   │   └── feature_builder.py   # Multivariate feature construction
│   ├── models/
│   │   ├── lstm_model.py     # LSTM architecture
│   │   └── trainer.py        # Training loop with early stopping
│   ├── evaluation/
│   │   └── evaluator.py      # Metrics, CV, plotting
│   └── utils/
│       ├── logger.py         # Logging configuration
│       ├── cache.py          # Disk-based caching
│       └── parallel.py       # Parallel processing
├── scripts/
│   ├── preprocess.py         # Video -> features pipeline
│   └── train.py              # Model training and evaluation
├── requirements.txt
└── README.md
```

## Installation

```bash
# Clone the repository
git clone https://github.com/VictorOlegario/deepfakeippg.git
cd deepfakeippg

# Install dependencies
pip install -r requirements.txt
```

## Dataset

This project uses the [FaceForensics++](https://github.com/ondyari/FaceForensics) dataset.

Expected directory structure:
```
data/raw/
├── original_sequences/
│   └── youtube/
│       └── c23/
│           └── videos/
│               ├── 000.mp4
│               └── ...
└── manipulated_sequences/
    ├── Deepfakes/c23/videos/
    ├── FaceSwap/c23/videos/
    ├── Face2Face/c23/videos/
    └── NeuralTextures/c23/videos/
```

## Usage

### 1. Preprocessing

Extract rPPG features from videos:

```bash
python scripts/preprocess.py \
    --data_dir data/raw \
    --output_dir data/processed \
    --target_fps 30.0 \
    --window_seconds 30.0 \
    --max_workers 4
```

Output: `X.npy` (features), `y.npy` (labels), `stats.json` (processing statistics)

### 2. Training

Train the LSTM model:

```bash
python scripts/train.py \
    --data_dir data/processed \
    --output_dir results \
    --epochs 100 \
    --batch_size 32 \
    --hidden_size 64 \
    --learning_rate 0.001
```

### 3. Cross-Validation

Run 5-fold stratified cross-validation:

```bash
python scripts/train.py \
    --data_dir data/processed \
    --output_dir results \
    --cross_validate \
    --n_folds 5
```

### 4. EDA Notebook

Explore signal characteristics:

```bash
jupyter notebook notebooks/eda_rppg_analysis.ipynb
```

## Methodology

### rPPG Signal Extraction

Three complementary methods extract blood volume pulse signals from facial video:

| Method | Description | Reference |
|--------|-------------|-----------|
| **GREEN** | Mean green channel intensity | Verkruysse et al. (2008) |
| **CHROM** | Chrominance-based projection | De Haan & Jeanne (2013) |
| **POS** | Plane orthogonal to skin tone | Wang et al. (2017) |

### Facial Regions of Interest

| ROI | Region | Rationale |
|-----|--------|-----------|
| R1 | Forehead (10-35% vertical) | Thin skin, superficial vessels |
| R2 | Left Cheek (45-75% vertical) | Rich blood supply |
| R3 | Right Cheek (45-75% vertical) | Symmetry analysis |

### Signal Processing

1. **Resample** to 30 Hz (cubic interpolation)
2. **Detrend** (remove linear trend)
3. **Normalize** (zero mean, unit variance)
4. **Bandpass filter** (0.7-3.0 Hz = 42-180 BPM, 4th order Butterworth)

### Model Architecture

```
Input (batch, seq_len, 9)
    |
Bidirectional LSTM (hidden=64, layers=2, dropout=0.3)
    |
Last hidden state (128-dim)
    |
FC (128 -> 32) + ReLU + Dropout(0.3)
    |
FC (32 -> 1) -> Sigmoid
    |
Output: P(fake)
```

## EDA Insights

### Key Findings

1. **Temporal consistency**: Real signals show stable, periodic heart rate patterns; deepfakes exhibit irregular waveforms
2. **Spectral clarity**: Real signals have sharp PSD peaks in the HR band (0.7-3.0 Hz); fake signals show broader, noisier spectra
3. **Spatial consistency**: Real videos show high inter-ROI correlation; deepfakes show significantly lower correlation between facial regions
4. **Signal quality**: Real signals have higher SNR and lower noise variance

### Method Comparison

- **POS**: Best separation between real/fake (most motion-robust)
- **CHROM**: Good balance of performance and computational cost
- **GREEN**: Simplest but most sensitive to noise
- **Combined**: All three together provide complementary discriminative features

### ROI Analysis

- **Forehead (R1)**: Cleanest signal (thinner skin, superficial blood vessels)
- **Cheeks (R2, R3)**: Valuable for spatial consistency analysis
- **All ROIs combined**: Best overall discrimination

## Performance Features

- **Parallel processing**: Multi-worker video processing via ProcessPoolExecutor
- **Disk caching**: Avoid reprocessing videos (SHA256-based cache keys)
- **Early stopping**: Prevent overfitting with patience-based training
- **Gradient clipping**: Stabilize LSTM training (max_norm=1.0)
- **LR scheduling**: ReduceLROnPlateau for adaptive learning rates

## Future Improvements

- [ ] Add attention mechanism to LSTM for interpretability
- [ ] Implement Transformer-based architecture
- [ ] Add cross-dataset evaluation (Celeb-DF, DFDC)
- [ ] Implement online/streaming inference
- [ ] Add GAN-based data augmentation
- [ ] Compare with CNN-based approaches (XceptionNet)
- [ ] Add simple baseline (Logistic Regression) for comparison

## References

1. Verkruysse, W., et al. (2008). "Remote plethysmographic imaging using ambient light." *Optics Express*.
2. De Haan, G., & Jeanne, V. (2013). "Robust pulse rate from chrominance-based rPPG." *IEEE TBME*.
3. Wang, W., et al. (2017). "Algorithmic principles of remote PPG." *IEEE TBME*.
4. Rossler, A., et al. (2019). "FaceForensics++: Learning to detect manipulated facial images." *ICCV*.
5. Ciftci, U. A., et al. (2020). "FakeCatcher: Detection of synthetic portrait videos using biological signals." *IEEE TPAMI*.

## License

This project is for research purposes only.
