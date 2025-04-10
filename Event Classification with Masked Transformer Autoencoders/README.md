### Data Analysis & Preparation
1. **Data Loading & Cleaning**:
   - Verified no null values (`isna().sum()` shows 0 nulls)

2. **Feature Analysis**:
   - Identified top correlated features with target (y):
     - Positive: x14, x6, x13, x1, x18, x17, x10, x21, x9
   - Generated correlation heatmap and feature-target bar plot
     ![Feature-Target Correlation.png](https://github.com/Abdelrahman10101/CMS/blob/main/Event%20Classification%20with%20Masked%20Transformer%20Autoencoders/Feature-Target%20Correlation.png)

### Modeling Approach
1. **Architecture**:
   - **Transformer Autoencoder** (TAE):
     - Input projection → Transformer encoder → Decoder
     - d_model=64, num_heads=4, num_layers=2, d_ff=256
   - **Classifier**:
     - 3-layer MLP (128 → 64 → 1) with ReLU and dropout
   - **Combined Model**: TAE reconstructions concatenated with original features

2. **Training Strategies**:
   - **Large Batch Size**: 8192 for GPU utilization and fast training
   - **Learning Rate**:
     - Initial LR: 0.001
     - ReduceLROnPlateau (min_lr=0.00025) when validation metric plateaus
   - **Early Stopping**: Patience=5 epochs
   - **Two-phase Training**:
     1. First train autoencoder (MSE loss)
     2. Then train classifier (BCE loss) using frozen TAE

### Results
- **Test AUC**: 0.8040 
  ![ROC Curve.png](https://github.com/Abdelrahman10101/CMS/blob/main/Event%20Classification%20with%20Masked%20Transformer%20Autoencoders/ROC%20Curve.png)


### Key Observations
1. **Feature Selection Attempts**:
   - Tried using only top correlated features → No AUC improvement
   - Attempted outlier handling (min and max bound) → No significant gain

2. **Anomoly Latent Detection**:
   - Tried Removing Anomoly Latent Detection → No AUC improvement
     ![Latent Space Projection](https://raw.githubusercontent.com/Abdelrahman10101/CMS/main/Event%20Classification%20with%20Masked%20Transformer%20Autoencoders/Latent%20Space%20Projection.png)

