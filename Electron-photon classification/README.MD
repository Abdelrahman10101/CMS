## Key Features and Approach

1. **Data Visualization and Intuition**:
   - Visualized electron and photon samples by plotting both energy deposit (channel 0) and hit time (channel 1) separately
   - Created RGB overlays combining both channels (energy in green, time in red) to understand feature relationships
   - Observed distinct patterns between electron and photon events in both energy distribution and timing
   - Electron:

![Electron Channels.png](https://github.com/Abdelrahman10101/CMS/blob/main/Electron-photon%20classification/Electron%20Channels.png)
![Electron.png](https://github.com/Abdelrahman10101/CMS/blob/main/Electron-photon%20classification/Electron.png)

   - Photon:

![Photon Channels.png](https://github.com/Abdelrahman10101/CMS/blob/main/Electron-photon%20classification/Photon%20Channels.png)
![Photon.png](https://github.com/Abdelrahman10101/CMS/blob/main/Electron-photon%20classification/Photon.png)

2. **Data Preparation**:
   - Loaded 249k samples each for electrons (label=0) and photons (label=1)
   - Split each dataset maintaining 80% for training and 20% for testing
   - Combined the splits to create balanced training and test sets
   - Converted to PyTorch tensors with shape (N, 2, 32, 32) for channel-first format

3. **Model Architecture**:
   - Implemented ResNet15 with residual blocks (simplified from standard ResNet)
   - Architecture:
     - Initial conv (2→64 channels)
     - 4 residual layers with [64, 128, 256, 512] channels
     - Adaptive average pooling
     - Final fully-connected layer

4. **Training Strategy**:
   - Used Adam optimizer with weight decay (1e-4)
   - Initial learning rate of 0.001
   - ReduceLROnPlateau scheduler (factor=0.5, patience=3)
   - Early stopping (patience=5) based on validation accuracy
   - Cross-entropy loss for this binary classification task

5. **Performance Metrics**:
   - Tracked both accuracy and AUC (Area Under ROC Curve)
   - Achieved final test AUC of 0.8104
   - Best validation accuracy of 0.7398
   - Confusion matrix shows balanced performance across classes
![ROC Curve.png](https://github.com/Abdelrahman10101/CMS/blob/main/Electron-photon%20classification/ROC%20Curve.png)
![Confussion Matrix.png](https://github.com/Abdelrahman10101/CMS/blob/main/Electron-photon%20classification/Confussion%20Matrix.png)
## Key Observations

1. **Training Dynamics**:
   - Model showed steady improvement in both training and validation metrics
   - Learning rate was reduced twice (0.001 → 0.0005 → 0.00025 → 0.000125 → 0.000063)
   - Early stopping wasn't triggered as performance kept improving

2. **Performance Analysis**:
   - The AUC score of 0.81 indicates good discriminative power
   - Confusion matrix shows no significant bias toward either class
   - Final performance suggests the model learned meaningful features from both energy and time channels

3. **Potential Improvements**:
   - Experiment with different architectures (deeper networks, attention mechanisms)
   - Try more aggressive data augmentation
   - Adjust class weights if imbalance is detected
   - Explore different learning rate schedules

The approach successfully distinguished between electron and photon events using their energy deposition and timing patterns, demonstrating the effectiveness of CNNs for this type of particle physics classification task.
