# Model Training Explanation

## 🎯 Your Questions Answered

### 1. How does the model get trained by data folder?

The training process works as follows:

#### 📁 Data Folder Structure
```
data/
├── valid_01.py    # Valid Python code
├── valid_02.py    # Valid Python code
├── ...
├── invalid_01.py  # Invalid Python code (syntax errors)
├── invalid_02.py  # Invalid Python code (syntax errors)
└── ...
```

- **20 valid Python files** (should parse successfully)
- **30 invalid Python files** (contain syntax errors)
- **Total: 50 training samples**

#### 🔄 Training Process Flow

1. **Load Files**: Read all `.py` files from `data/` folder
2. **Parse Code**: Use Tree-sitter to parse each file
3. **Create Labels**: 
   - `0` = Valid code (parses successfully)
   - `1` = Invalid code (has syntax errors)
4. **Convert to Graphs**: Transform valid code into graph representations
5. **Train GNN**: Feed graphs to Graph Convolutional Network

### 2. How are Python files converted to graphs?

#### 🌳 AST → Graph Conversion

**Step 1: Parse Code**
```python
# Example: valid_01.py
def add(a, b):
    return a + b
```

**Step 2: Create AST (Abstract Syntax Tree)**
```
module
├── function_definition
│   ├── identifier (add)
│   ├── parameters
│   │   ├── identifier (a)
│   │   └── identifier (b)
│   └── block
│       └── return_statement
│           └── binary_operator
│               ├── identifier (a)
│               └── identifier (b)
```

**Step 3: Convert AST to Graph**
- **Nodes**: Each AST node becomes a graph node
- **Edges**: Parent-child relationships become graph edges
- **Features**: Each node has 3 features:
  - `[type_id, depth, num_children]`

#### 📊 Graph Structure Example

For the code `def add(a, b): return a + b`:

```
Graph:
- Nodes: 12
- Features per node: 3
- Edges: 11 (parent-child connections)
- Node features: [type_index, depth, children_count]
```

### 3. How does training work with graphs?

#### 🧠 GNN Training Process

**Step 1: Graph Creation**
```python
# Valid code → Real graph with features
valid_graph = ast_to_graph(valid_code)
# Invalid code → Dummy graph (single node)
invalid_graph = Data(x=[[0,0,0]], edge_index=[])
```

**Step 2: Labeling**
```python
valid_graph.y = torch.tensor([0])    # Valid = 0
invalid_graph.y = torch.tensor([1])  # Invalid = 1
```

**Step 3: Batching**
```python
# Multiple graphs are batched together
train_loader = DataLoader(graphs, batch_size=8)
```

**Step 4: GNN Forward Pass**
```python
# Model processes graph structure
output = model(graph_batch)
# Output: [batch_size, 2] (probabilities for valid/invalid)
```

#### 🎯 What the Model Learns

The GNN learns to:
- **Recognize valid syntax patterns** in graph structure
- **Detect invalid syntax patterns** in graph structure
- **Classify graphs** as valid (0) or invalid (1)
- **Provide confidence scores** for predictions

### 4. Training Data Flow

#### 📂 File Loading
```python
# train_model.py - load_dataset()
for fname in os.listdir(data_dir):
    with open(fname, 'r') as f:
        code = f.read()
    
    # Parse with Tree-sitter
    root = parse_code(code)
    if root is None:
        labels.append(1)  # Invalid
    else:
        labels.append(0)  # Valid
```

#### 🕸️ Graph Building
```python
# train_model.py - build_graphs()
for code in samples:
    if code is None:
        # Create dummy graph for invalid code
        graph = Data(x=[[0,0,0]], edge_index=[])
    else:
        # Create real graph for valid code
        graph = ast_to_graph(code)
```

#### 🧠 Model Training
```python
# train_model.py - train_model()
for epoch in range(epochs):
    for batch in train_loader:
        # Forward pass
        output = model(batch)
        loss = criterion(output, batch.y)
        
        # Backward pass
        loss.backward()
        optimizer.step()
```

### 5. Key Insights

#### ✅ What Works Well
- **Tree-sitter parsing** correctly identifies syntax errors
- **Graph conversion** preserves code structure
- **GNN learns** to distinguish valid/invalid patterns
- **BatchNorm issue** (needs multiple samples per batch)

#### 🔧 Training Process Summary
1. **Data Loading**: Read 50 Python files from `data/` folder
2. **Parsing**: Use Tree-sitter to check syntax validity
3. **Graph Creation**: Convert valid code to graphs, invalid to dummy graphs
4. **Labeling**: 0=valid, 1=invalid
5. **Training**: GNN learns from graph structure patterns
6. **Saving**: Best model saved as `syntax_error_model.pth`

#### 🎯 The Model Learns
- **Valid syntax** → Complex graph structure → Predict 0
- **Invalid syntax** → Dummy graph → Predict 1
- **Pattern recognition** in AST structure
- **Confidence scoring** for predictions

### 6. File Examples

#### Valid File (`valid_01.py`)
```python
def add(a, b):
    return a + b
```
**Result**: ✅ Parses successfully → Real graph → Label 0

#### Invalid File (`invalid_01.py`)
```python
def test()
    pass
```
**Result**: ❌ Syntax error → Dummy graph → Label 1

### 7. Training Output

The training process produces:
- **`syntax_error_model.pth`**: Trained GNN model
- **`training_history.json`**: Training metrics
- **Model learns** to classify code as valid/invalid based on graph structure

---

## 🎯 Summary

**The model gets trained by:**
1. Reading Python files from `data/` folder
2. Parsing each file with Tree-sitter
3. Converting valid code to graphs, invalid code to dummy graphs
4. Training a GNN to classify graphs as valid (0) or invalid (1)
5. Learning patterns in AST structure to detect syntax errors

**The training data consists of:**
- 20 valid Python files (complex graphs)
- 30 invalid Python files (dummy graphs)
- Total: 50 training samples

**The model learns to:**
- Recognize valid syntax patterns in graph structure
- Detect invalid syntax patterns
- Provide confidence scores for predictions 