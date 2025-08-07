# Python Syntax Error Classifier

A Graph Neural Network (GNN) based system for detecting and classifying Python syntax errors. This project uses a clean, focused approach without regex fallbacks, with dynamic data generation to handle overfitting/underfitting.

## 🚀 Features

- **Pure GNN Architecture**: No regex fallbacks - relies entirely on graph neural networks
- **Dynamic Data Generation**: Automatically generates training data based on model performance
- **Overfitting/Underfitting Detection**: Monitors training progress and adjusts accordingly
- **6 Error Types**: Classifies syntax errors into 6 distinct categories
- **High Accuracy**: Achieves high accuracy on syntax error detection
- **Clean Architecture**: Streamlined codebase without unnecessary complexity

## 📋 Error Types Detected

1. **valid** - Syntactically correct Python code
2. **missing_colon** - Missing colon after control flow statements
3. **unclosed_string** - Unclosed string literals
4. **unexpected_indent** - Unexpected indentation without proper context
5. **unexpected_eof** - Unexpected end of file (unclosed brackets, parentheses, etc.)
6. **invalid_token** - Invalid characters or tokens

## 🏗️ Architecture

### Model Structure
```
Input Code → Tokenization → Graph Construction → GNN Processing → Classification
```

### GNN Architecture
- **3 Graph Convolution Layers** (GCNConv)
- **Batch Normalization** after each conv layer
- **Dropout** (0.3) for regularization
- **Global Pooling** (mean + max)
- **3 Linear Layers** for classification
- **ReLU Activation** throughout

### Key Components

1. **TokenBasedGraphBuilder**: Converts Python code to graph representation
2. **SyntaxErrorGNN**: Graph Neural Network for classification
3. **SyntaxErrorClassifier**: Main classifier interface
4. **DynamicDataGenerator**: Generates training data dynamically
5. **ModelTrainer**: Handles training with overfitting detection

## 🛠️ Installation

### Prerequisites
- Python 3.8+
- PyTorch 1.9+
- PyTorch Geometric 2.0+

### Install Dependencies
```bash
pip install torch torch-geometric scikit-learn numpy
```

## 📚 Usage

### Training the Model

```bash
python train_model.py
```

This will:
- Generate balanced training data (100 samples per class)
- Train the GNN model with overfitting detection
- Save the trained model as `syntax_error_model.pth`
- Save training history as `training_history.json`

### Using the Classifier

```python
from syntax_error_classifier import SyntaxErrorClassifier

# Initialize classifier
classifier = SyntaxErrorClassifier()

# Analyze code
result = classifier.analyze_code('def test()\n    pass')
print(f"Error: {result['result']}")
print(f"Confidence: {result['confidence']:.2f}")
```

### Testing Scripts

#### Quick Demo
```bash
python demo.py
```
Shows the model working with various examples and provides accuracy metrics.

#### Quick Test
```bash
python quick_test.py
```
Runs predefined test cases to verify model performance.

#### Interactive Testing
```bash
python interactive_test.py
```
Choose from:
1. Interactive mode (input code manually)
2. Batch testing (from file)
3. Demo mode

#### Piped Input Testing
```bash
echo "def test()\n    pass" | python pipe_test.py
```
Test with piped input (handles newlines properly).

#### Command Line Testing
```bash
python test_classifier.py "def test():\n    pass"
```
Test with command line arguments.

## 🔧 Training Features

### Dynamic Data Generation
- Automatically generates balanced datasets
- Creates additional samples for underperforming classes
- Handles data augmentation for better generalization

### Overfitting/Underfitting Detection
- Monitors training and validation losses
- Detects overfitting patterns (training loss ↓, validation loss ↑)
- Detects underfitting patterns (both losses high and stagnant)
- Automatically adjusts regularization when needed

### Training Process
1. **Data Generation**: Creates 600 samples (100 per class)
2. **Data Splitting**: 70% train, 15% validation, 15% test
3. **Training**: 100 epochs with early stopping
4. **Monitoring**: Checks for overfitting every 10 epochs
5. **Saving**: Best model saved based on validation accuracy

## 📊 Performance

### Expected Results
- **Overall Accuracy**: >85%
- **Per-class Accuracy**: >80% for each error type
- **Training Time**: ~5-10 minutes on CPU
- **Inference Time**: <1 second per code snippet

### Model Evaluation
The model provides:
- Classification report with precision, recall, F1-score
- Confusion matrix
- Per-class accuracy breakdown
- Confidence scores for predictions

## 🎯 Example Usage

```python
from syntax_error_classifier import SyntaxErrorClassifier

classifier = SyntaxErrorClassifier()

# Test cases
test_cases = [
    'def test():\n    pass',           # valid
    'def test()\n    pass',            # missing_colon
    'print("hello',                    # unclosed_string
    '  print("indented")',             # unexpected_indent
    'x = [1, 2, 3',                   # unexpected_eof
    'x = @invalid'                     # invalid_token
]

for code in test_cases:
    result = classifier.analyze_code(code)
    print(f"Code: {repr(code)}")
    print(f"Predicted: {result['result']} (confidence: {result['confidence']:.2f})")
    print("-" * 40)
```

## 📁 Project Structure

```
syntax-err/
├── syntax_error_classifier.py    # Main classifier
├── train_model.py               # Training script
├── demo.py                      # Demo with examples
├── quick_test.py                # Quick test cases
├── interactive_test.py          # Interactive testing
├── pipe_test.py                 # Piped input testing
├── test_classifier.py           # Command line testing
├── test_model_accuracy.py      # Accuracy testing
├── syntax_error_model.pth      # Trained model
├── training_history.json        # Training history
├── requirements.txt             # Dependencies
└── README.md                   # This file
```

## 🔍 Key Features Explained

### No Regex Fallbacks
- Pure GNN approach for better generalization
- Learns patterns from graph structure
- More robust than rule-based systems

### Dynamic Data Generation
- Generates training data programmatically
- Balances dataset across all error types
- Creates additional samples for problematic classes

### Overfitting Detection
- Monitors training vs validation metrics
- Automatically increases dropout when overfitting detected
- Provides recommendations for model improvement

### Clean Architecture
- Single responsibility classes
- No unnecessary complexity
- Easy to understand and modify

## 🚨 Error Handling

The model handles various edge cases:
- Empty code
- Very long code snippets
- Malformed input
- Missing model files

All errors are logged with clear messages and the system fails gracefully.

## 📈 Training Monitoring

During training, you'll see:
- Epoch progress with loss and accuracy
- Overfitting/underfitting warnings
- Model saving notifications
- Final evaluation results

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📄 License

This project is open source and available under the MIT License.

## 🆘 Troubleshooting

### Common Issues

1. **Model not loading**: Ensure `syntax_error_model.pth` exists
2. **CUDA errors**: Model automatically falls back to CPU
3. **Memory issues**: Reduce batch size in training
4. **Poor accuracy**: Retrain with more data or adjust hyperparameters

### Performance Tips

- Use GPU for faster training (automatically detected)
- Increase `samples_per_class` for better accuracy
- Adjust learning rate if training is unstable
- Monitor training history for insights

## 📞 Support

For issues and questions:
1. Check the troubleshooting section
2. Review the training logs
3. Test with the interactive mode
4. Create an issue with detailed information

---

**Note**: This model is designed for Python syntax error detection and may not work well with other programming languages or very complex code structures. 