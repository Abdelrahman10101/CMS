### Data Analysis & Preparation
1. **Data Loading & Cleaning**:
   - Loaded train/test parquet files (1.1M samples each)
   - Verified no null values (`isna().sum()` shows 0 nulls)
   - Renamed columns to standardized format (y, x1-x21)

2. **Feature Analysis**:
   - Identified top correlated features with target (y):
     - Positive: x14, x6, x13, x1, x18, x17, x10, x21, x9
   - Generated correlation heatmap and feature-target bar plot
   - Basic statistics show well-scaled features (means ~0, stds ~1)

### Modeling Approach
1. **Architecture**:
   - **Transformer Autoencoder** (TAE):
     - Input projection → Transformer encoder → Decoder
     - d_model=64, num_heads=4, num_layers=2, d_ff=256
   - **Classifier**:
     - 3-layer MLP (128 → 64 → 1) with ReLU and dropout
   - **Combined Model**: TAE reconstructions concatenated with original features

2. **Training Strategies**:
   - **Large Batch Size**: 8192 for GPU utilization
   - **Learning Rate**:
     - Initial LR: 0.001
     - ReduceLROnPlateau (min_lr=0.00025) when validation metric plateaus
   - **Early Stopping**: Patience=5 epochs
   - **Two-phase Training**:
     1. First train autoencoder (MSE loss)
     2. Then train classifier (BCE loss) using frozen TAE

### Results
- **Best Validation AUC**: 0.7936
- **Test AUC**: 0.7936 (no label bias)
- **Accuracy**: 72.07%
- Training showed consistent improvement up to 100 epochs

### Key Observations
1. **Feature Selection Attempts**:
   - Tried using only top correlated features → No AUC improvement
   - Attempted outlier removal → No significant gain

2. **Autoencoder Behavior**:
   - Reconstruction likely captures latent patterns
   - Combined features (original + reconstructed) boosted performance
