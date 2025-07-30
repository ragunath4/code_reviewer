# Quick Reference Guide - Syntax Error Detection Project

## 🚀 **Quick Start**

### **1. Setup Project**

```bash
# Install dependencies
pip install -r requirements.txt

# Create expanded dataset
python expanded_dataset.py

# Run improved training
python improved_trainer.py

# Compare models
python model_comparison.py
```

### **2. Generate Visual Diagrams**

```bash
python VISUAL_ARCHITECTURE.py
```

---

## 📋 **File Purpose Summary**

| File                  | Purpose                                        | Key Function                              |
| --------------------- | ---------------------------------------------- | ----------------------------------------- |
| `expanded_dataset.py` | Creates 50 code samples (20 valid, 30 invalid) | `create_expanded_dataset()`               |
| `improved_trainer.py` | Enhanced training with cross-validation        | `ImprovedTrainer.cross_validation()`      |
| `enhanced_model.py`   | Advanced model architectures                   | `EnhancedSyntaxGCN`, `AttentionSyntaxGCN` |
| `model_comparison.py` | Compares different model architectures         | `ModelComparator.compare_models()`        |
| `parser_util.py`      | Parses Python code using Tree-sitter           | `parse_code(code)`                        |
| `graph_builder.py`    | Converts AST to graph representation           | `ast_to_graph(code)`                      |
| `model.py`            | Original simple GCN model                      | `SyntaxGCN`                               |

---

## 🔄 **Core Data Flow**

```
Code Sample → parser_util.py → AST → graph_builder.py → Graph → enhanced_model.py → Prediction
```

### **Step-by-Step Process:**

1. **Input**: Python code string
2. **Parsing**: `parser_util.py` uses Tree-sitter to create AST
3. **Conversion**: `graph_builder.py` converts AST to PyTorch Geometric graph
4. **Features**: Each node gets features `[type_id, depth, num_children]`
5. **Processing**: GCN model processes graph structure
6. **Output**: Binary classification (0=valid, 1=invalid)

---

## 🧠 **Key Concepts Explained**

### **AST (Abstract Syntax Tree)**

```
Code: def test(): return 42
AST:  function_def
       ├── name: "test"
       ├── parameters: []
       └── body: return_statement
                 └── value: 42
```

### **Graph Representation**

- **Nodes**: AST nodes (functions, classes, variables, etc.)
- **Edges**: Parent-child relationships in AST
- **Features**: `[type_id, depth, num_children]`

### **GCN (Graph Convolutional Network)**

- Processes graph structure
- Learns node representations
- Aggregates information globally
- Classifies entire graph

---

## 📊 **Model Architectures**

| Model                | Description                      | Expected Accuracy |
| -------------------- | -------------------------------- | ----------------- |
| `SyntaxGCN`          | Original simple model            | ~85%              |
| `EnhancedSyntaxGCN`  | Multiple layers + regularization | ~90%              |
| `AttentionSyntaxGCN` | Multi-head attention             | ~92%              |
| `ResidualSyntaxGCN`  | Residual connections             | ~91%              |

---

## 🔧 **Configuration Options**

### **Dataset Size** (`expanded_dataset.py`)

```python
valid_samples = 20    # Number of valid code samples
invalid_samples = 30  # Number of invalid code samples
```

### **Model Parameters** (`enhanced_model.py`)

```python
hidden_dim = 64       # Hidden layer size
num_layers = 3        # Number of GCN layers
dropout = 0.3         # Dropout rate
```

### **Training Parameters** (`improved_trainer.py`)

```python
batch_size = 8        # Batch size
lr = 0.001           # Learning rate
patience = 15        # Early stopping patience
n_folds = 5          # Cross-validation folds
```

---

## 📈 **Expected Results**

### **Cross-Validation Results**

```
train_acc: 0.9500 ± 0.0200
val_acc:   0.9200 ± 0.0300
test_acc:  0.9000 ± 0.0400
```

### **Model Comparison**

```
OriginalSyntaxGCN:     Test Acc: 0.8500, F1: 0.8400
EnhancedSyntaxGCN:     Test Acc: 0.9000, F1: 0.8900
AttentionSyntaxGCN:    Test Acc: 0.9200, F1: 0.9100
ResidualSyntaxGCN:     Test Acc: 0.9100, F1: 0.9000
```

---

## 🐛 **Common Issues & Solutions**

### **1. Tree-sitter Import Error**

```bash
cd tree-sitter-python
python setup.py build
```

### **2. Memory Issues**

```python
# Reduce batch size
batch_size = 4  # Instead of 8
```

### **3. Training Not Converging**

```python
# Reduce learning rate
lr = 0.0001  # Instead of 0.001
```

### **4. Overfitting**

```python
# Increase dropout
dropout = 0.5  # Instead of 0.3
```

---

## 🎯 **Key Functions to Understand**

### **Code Parsing**

```python
from parser_util import parse_code
root = parse_code("def test(): pass")
# Returns AST tree or None (if syntax error)
```

### **Graph Building**

```python
from graph_builder import ast_to_graph
graph = ast_to_graph("def test(): pass")
# Returns PyTorch Geometric Data object
```

### **Model Prediction**

```python
from enhanced_model import EnhancedSyntaxGCN
model = EnhancedSyntaxGCN(num_node_types=len(NODE_TYPE_TO_IDX))
prediction = model(graph)  # Returns logits
```

---

## 📁 **Output Files**

| File                            | Description                       |
| ------------------------------- | --------------------------------- |
| `syntax_error_model.pth`        | Trained model weights             |
| `model_comparison_results.json` | Model comparison results          |
| `model_comparison.png`          | Visualization of model comparison |
| `project_architecture.png`      | Complete project architecture     |
| `simplified_flow.png`           | Simplified data flow              |

---

## 🔍 **Debugging Tips**

### **1. Check Data Loading**

```python
# Verify dataset creation
python expanded_dataset.py
ls data/  # Should show 50 .py files
```

### **2. Test Individual Components**

```python
# Test parsing
python -c "from parser_util import parse_code; print(parse_code('def test(): pass'))"

# Test graph building
python -c "from graph_builder import ast_to_graph; print(ast_to_graph('def test(): pass'))"
```

### **3. Monitor Training**

```python
# Check training logs
# Look for early stopping messages
# Monitor validation accuracy
```

---

## 🚀 **Next Steps**

1. **Run the complete system**: `python improved_trainer.py`
2. **Compare models**: `python model_comparison.py`
3. **Generate diagrams**: `python VISUAL_ARCHITECTURE.py`
4. **Analyze results**: Check generated files
5. **Customize**: Modify parameters in respective files
6. **Scale up**: Add more code samples to dataset

---

## 💡 **Understanding Tips**

1. **Start with the data flow**: Understand how code becomes graphs
2. **Focus on one component at a time**: Don't try to understand everything at once
3. **Use the visual diagrams**: They show the big picture
4. **Experiment with parameters**: See how changes affect results
5. **Check the logs**: They provide insights into what's happening

This project demonstrates:

- ✅ Graph Neural Networks for code analysis
- ✅ AST-based feature extraction
- ✅ Cross-validation for robust evaluation
- ✅ Multiple model architectures
- ✅ Comprehensive error handling
- ✅ Scalable and modular design
