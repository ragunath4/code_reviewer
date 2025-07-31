# Python Syntax Error Detector

A machine learning-based tool for detecting Python syntax errors using Graph Convolutional Networks (GCN) and Tree-sitter parsing.

## 🏗️ Project Structure

```
syntax-err/
├── src/                    # Source code
│   ├── core/              # Core components
│   │   ├── parser_util.py      # Tree-sitter parser utilities
│   │   └── graph_builder.py    # AST to graph conversion
│   ├── models/            # Neural network models
│   │   └── enhanced_model.py   # GCN model implementation
│   └── analysis/          # Analysis tools
│       └── syntax_analyzer_gcn_only.py  # Main analyzer
├── data/                  # Training and test data
│   ├── valid/            # Valid Python code samples
│   └── invalid/          # Invalid Python code samples
├── tests/                # Test files
├── docs/                 # Documentation
├── examples/             # Example usage
├── main.py               # Main entry point
└── requirements.txt      # Dependencies
```

## 🚀 Quick Start

### Installation

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd syntax-err
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Train the model (if needed):**
   ```bash
   python train_model.py
   ```

### Usage

#### Analyze a Python file:
```bash
python main.py --file data/valid/valid_01.py
```

#### Analyze code from command line:
```bash
python main.py --code "def test(): pass"
```

#### Analyze with custom model:
```bash
python main.py --file data/valid/valid_01.py --model path/to/model.pth
```

## 📊 Features

- **Tree-sitter Parsing**: Robust Python syntax parsing
- **GCN Analysis**: Graph Convolutional Network for complex pattern detection
- **Incomplete Code Detection**: Identifies incomplete function/class definitions
- **Confidence Scoring**: Provides confidence levels for predictions
- **Error Classification**: Categorizes different types of syntax errors
- **Detailed Recommendations**: Suggests fixes for detected errors

## 🔧 Core Components

### Parser (`src/core/parser_util.py`)
- Uses Tree-sitter for Python parsing
- Recursive error detection in AST
- Handles both valid and invalid code

### Graph Builder (`src/core/graph_builder.py`)
- Converts AST to graph representation
- Extracts node features (type, depth, children count)
- Creates edge connections between nodes

### GCN Model (`src/models/enhanced_model.py`)
- Graph Convolutional Network architecture
- Processes graph-structured code data
- Outputs binary classification (valid/invalid)

### Analyzer (`src/analysis/syntax_analyzer_gcn_only.py`)
- Main analysis engine
- Combines parser and GCN predictions
- Provides detailed error reports

## 📈 Confidence Levels

- **Valid Code**: 80% confidence
- **Invalid Code**: 95-98% confidence
- **Incomplete Code**: 95% confidence
- **Parsing Errors**: 98% confidence

## 🧪 Testing

### Test valid code:
```bash
python main.py --file data/valid/valid_01.py
```

### Test invalid code:
```bash
python main.py --file data/invalid/invalid_01.py
```

### Test incomplete code:
```bash
python main.py --code "def test():"
```

## 📝 Error Types

1. **Parsing Failed**: Code cannot be parsed by Tree-sitter
2. **Incomplete Code**: Missing method bodies or incomplete structures
3. **GCN Invalid**: GCN model detected potential issues
4. **Valid Syntax**: Code is syntactically correct

## 🔍 Analysis Examples

### Valid Code
```python
def add(a, b):
    return a + b
```
**Result**: ✅ VALID SYNTAX (80% confidence)

### Invalid Code
```python
def test()
    pass
```
**Result**: ❌ SYNTAX ERROR (98% confidence) - Missing colon

### Incomplete Code
```python
class Calculator:
    def __init__(self):
```
**Result**: ❌ SYNTAX ERROR (95% confidence) - Incomplete method

## 🛠️ Development

### Project Organization
- **Modular Design**: Each component is in its own module
- **Clean Imports**: Proper package structure with `__init__.py` files
- **Separation of Concerns**: Parser, model, and analysis are separate

### Adding New Features
1. Add new components to appropriate `src/` subdirectory
2. Update `__init__.py` files for imports
3. Add tests in `tests/` directory
4. Update documentation

## 📚 Dependencies

- **PyTorch**: Deep learning framework
- **PyTorch Geometric**: Graph neural networks
- **Tree-sitter**: Fast parsing
- **NumPy**: Numerical computations
- **Scikit-learn**: Machine learning utilities

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License.

## 🆘 Support

For issues and questions:
1. Check the documentation in `docs/`
2. Review example usage in `examples/`
3. Open an issue on GitHub 