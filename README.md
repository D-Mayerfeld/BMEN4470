# TCN Demo: From Basics to ECG Classification

This repository contains a comprehensive demonstration of Temporal Convolutional Networks (TCNs) applied to ECG classification for cardiac conditions. The demo includes both theoretical foundations and practical implementations with automatic configuration comparisons and real-time visualization.

## 🚀 Features

### 🧠 TCN Architecture
- **Causal Convolutions**: Prevent information leakage from future timesteps
- **Dilated Convolutions**: Exponentially increase receptive field (1, 2, 4, 8...)
- **Residual Connections**: Skip connections to enable deeper networks and better gradient flow
- **Hierarchical Processing**: Extract features at multiple temporal scales
- **Weight Normalization**: Stabilize training dynamics

### 📊 ECG Data Generation
- **4 Cardiac Conditions**: Normal, Atrial Fibrillation, Bradycardia, Tachycardia
- **Realistic ECG Signals**: Synthetic data with proper heart rates and noise
- **Balanced Dataset**: Equal samples per class with proper randomization
- **Signal Visualization**: Examples of each cardiac condition

### 🎨 Architecture Visualization
- **Hierarchical Diagrams**: CNN-style TCN architecture visualization
- **Temporal Flow**: Shows causal convolution connections
- **Receptive Field**: Visual representation of temporal coverage
- **Parameter Counting**: Accurate parameter calculation with/without residuals
- **Dilation Patterns**: Clear visualization of exponential dilation

### 📈 Automatic Demo Configurations
- **Configuration 1**: Small kernel (k=2), shallow network [32, 32]
- **Configuration 2**: Larger kernel (k=3), deeper network [64, 64, 64]
- **Configuration 3**: Large kernel (k=5), very deep network [128, 128, 128, 128]
- **Residual Comparison**: Direct performance comparison WITH vs WITHOUT residual connections
- **Performance Analysis**: Side-by-side configuration comparison

### 🔧 Training Features
- **Automatic Execution**: All demos run automatically when notebook is executed
- **Real-time Visualization**: Training curves, confusion matrices, sample predictions
- **Debug Information**: Training diagnostics and validation accuracy tracking
- **ECG Signal Examples**: Visual examples of generated cardiac conditions
- **Class Distribution**: Balanced dataset verification

## 📁 Files

- `TCN_Interactive_Demo.ipynb`: **Main Jupyter notebook** with all features and automatic demos
- `requirements.txt`: Python dependencies
- `README.md`: This documentation

## 🚀 Quick Start

### Prerequisites
```bash
# Install dependencies
pip install -r requirements.txt
```

### Running the Demo
```bash
# Start Jupyter
jupyter notebook

# Open TCN_Interactive_Demo.ipynb
# Run all cells to see all demos automatically
```

### What Happens When You Run the Notebook
1. **Setup**: Imports libraries and defines TCN architecture
2. **ECG Generation**: Creates synthetic ECG dataset with examples
3. **Architecture Visualization**: Shows TCN structure and statistics
4. **Demo 1**: Small kernel, shallow network training
5. **Demo 2**: Larger kernel, deeper network training
6. **Demo 3**: Large kernel, very deep network training
7. **Residual Comparison**: WITH vs WITHOUT residual connections
8. **Performance Analysis**: Side-by-side comparison of all configurations

## 📚 What You'll Learn

### 1. TCN Fundamentals
- **Causal Convolutions**: How temporal causality is maintained
- **Dilated Convolutions**: Exponential dilation pattern (1, 2, 4, 8...)
- **Residual Connections**: Skip connections for deeper networks
- **Receptive Field**: How temporal coverage increases with depth

### 2. Architecture Design
- **Kernel Size Effects**: Small vs large kernels for temporal patterns
- **Network Depth**: Shallow vs deep networks and capacity
- **Residual Impact**: Performance improvement from skip connections
- **Parameter Efficiency**: Balancing model size and performance

### 3. ECG Signal Processing
- **Cardiac Conditions**: Normal, AFib, Bradycardia, Tachycardia characteristics
- **Temporal Patterns**: Heart rate variations and rhythm irregularities
- **Feature Extraction**: How TCNs capture temporal ECG features
- **Classification**: Multi-class cardiac condition detection

### 4. Training Dynamics
- **Loss Curves**: Training and validation loss evolution
- **Accuracy Tracking**: Learning progress monitoring
- **Gradient Flow**: How residual connections help training
- **Convergence Patterns**: Different architecture behaviors

## 🎯 Key Insights

### Kernel Size Effects
- **Small kernels (k=2)**: Fast training, local patterns, fewer parameters
- **Large kernels (k=5)**: Longer dependencies, more parameters, slower training
- **Optimal choice**: Depends on signal complexity and computational budget

### Residual Connections
- **WITH Residuals**: Better gradient flow, deeper networks, improved performance
- **WITHOUT Residuals**: Simpler architecture, potential vanishing gradients
- **Performance Impact**: Typically 5-15% accuracy improvement with residuals

### Network Depth
- **Shallow [32, 32]**: Fast training, may underfit complex patterns
- **Deep [128, 128, 128, 128]**: More capacity, risk of overfitting
- **Sweet spot**: Balance between capacity and generalization

## 🔧 Technical Implementation

### TCN Architecture
```python
class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding):
        # Causal convolutions with dilation
        # Residual connections (skip connections)
        # ReLU activations
        # Chomp operation (remove future padding)
```

### ECG Generation
- **Normal**: Regular rhythm, 70-85 BPM, consistent QRS complexes
- **Atrial Fibrillation**: Irregular rhythm, 120-160 BPM, variable RR intervals
- **Bradycardia**: Slow rhythm, 45-55 BPM, normal QRS morphology
- **Tachycardia**: Fast rhythm, 120-150 BPM, compressed T-waves

### Training Pipeline
1. **Data Generation**: 1000 synthetic ECG samples (250 per class)
2. **Data Splitting**: 80% train, 20% validation with stratification
3. **Model Training**: Adam optimizer, CrossEntropy loss
4. **Evaluation**: Accuracy, confusion matrix, sample predictions

## 🎓 Educational Value

### For Students
- **Visual Learning**: See TCN architecture and training in action
- **Configuration Comparison**: Understand how hyperparameters affect performance
- **Medical AI**: Learn about ECG signal processing and cardiac conditions
- **Deep Learning**: Understand temporal modeling and residual connections

### For Researchers
- **Architecture Comparison**: Systematic evaluation of TCN variants
- **Performance Analysis**: Detailed metrics and visualization
- **Code Structure**: Clean, well-documented implementation
- **Extensibility**: Easy to modify for different tasks

## 📊 Expected Results

### Typical Performance
- **Random Baseline**: ~25% accuracy (4 classes)
- **Good TCN Performance**: 60-90% validation accuracy
- **Residual Improvement**: 5-15% accuracy boost
- **Training Time**: 1-5 minutes per configuration

### Configuration Comparison
- **Configuration 1 (k=2, [32,32])**: Fast training, moderate performance
- **Configuration 2 (k=3, [64,64,64])**: Balanced training time and performance
- **Configuration 3 (k=5, [128,128,128,128])**: Slower training, higher performance
- **Residual vs No Residual**: Clear performance difference

## 🛠️ Troubleshooting

### Common Issues
1. **Multiple Training Runs**: This is expected - the notebook runs multiple configurations automatically
2. **Memory Issues**: Reduce batch size or number of epochs
3. **Slow Training**: Use smaller networks or fewer epochs for testing
4. **Widget Issues**: The notebook focuses on automatic demos, not interactive widgets

### Performance Tips
- **GPU Acceleration**: Automatically uses CUDA if available
- **Batch Size**: Adjust based on available memory
- **Epochs**: Start with fewer epochs for quick testing
- **Network Size**: Use smaller networks for faster experimentation

## 🔮 Extensions

### Possible Enhancements
1. **Attention Mechanisms**: Add self-attention to TCN layers
2. **Multi-scale Features**: Combine different kernel sizes
3. **Real ECG Data**: Apply to actual cardiac recordings
4. **More Conditions**: Add additional cardiac arrhythmias
5. **Ensemble Methods**: Combine multiple TCN models

### Advanced Features
1. **Hyperparameter Optimization**: Grid search, random search
2. **Data Augmentation**: Improve generalization
3. **Regularization**: Dropout, weight decay, batch normalization
4. **Transfer Learning**: Pre-train on large ECG datasets

## 📋 Requirements

- **Python**: 3.7+
- **PyTorch**: 1.12+
- **NumPy**: For numerical computations
- **Matplotlib**: For visualization
- **Seaborn**: For statistical plots
- **Scikit-learn**: For data splitting and metrics
- **Jupyter Notebook**: For interactive environment

## 📄 License

This educational demo is provided for learning purposes. Feel free to use and modify for your own projects and research.

## 🤝 Contributing

Contributions are welcome! Areas for improvement:
- Additional loss functions and optimizers
- More ECG conditions and realistic data
- Real dataset integration
- Performance optimizations
- Documentation enhancements
- Interactive visualization improvements

---

**Happy Learning!** 🚀

This demo provides a comprehensive understanding of TCNs from theoretical foundations to practical applications in medical signal processing, with automatic comparison of different architecture configurations and training dynamics.