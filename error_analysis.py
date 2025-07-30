import os
import torch
import numpy as np
from torch_geometric.data import DataLoader, Data
from graph_builder import ast_to_graph, NODE_TYPE_TO_IDX
from model import SyntaxGCN
from parser_util import parse_code
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

DATA_DIR = 'data'

def load_dataset():
    samples = []
    labels = []
    filenames = []
    for fname in os.listdir(DATA_DIR):
        if not fname.endswith('.py'):
            continue
        with open(os.path.join(DATA_DIR, fname), 'r', encoding='utf-8') as f:
            code = f.read()
        root = parse_code(code)
        if root is None:
            samples.append(None)
            labels.append(1)  # Syntax error
        else:
            samples.append(code)
            labels.append(0)  # Valid
        filenames.append(fname)
    return samples, labels, filenames

def build_graphs(samples):
    graphs = []
    for code in samples:
        if code is None:
            # Dummy graph for syntax error
            x = torch.tensor([[0, 0, 0]], dtype=torch.float)
            edge_index = torch.zeros((2, 0), dtype=torch.long)
            graphs.append(Data(x=x, edge_index=edge_index))
        else:
            g = ast_to_graph(code)
            graphs.append(g)
    return graphs

def analyze_model_performance():
    print("=== MODEL PERFORMANCE ANALYSIS ===\n")
    
    # Load data
    samples, labels, filenames = load_dataset()
    
    # Build node type dict
    for code in samples:
        if code is not None:
            root = parse_code(code)
            from graph_builder import build_node_type_dict
            build_node_type_dict(root)
    
    graphs = build_graphs(samples)
    for i, g in enumerate(graphs):
        g.y = torch.tensor([labels[i]], dtype=torch.long)
        g.batch = torch.zeros(g.x.size(0), dtype=torch.long)
    
    # Train-test split
    X_train, X_test, y_train, y_test, train_files, test_files = train_test_split(
        graphs, labels, filenames, test_size=0.3, random_state=42, stratify=labels)
    
    print(f"Dataset size: {len(samples)}")
    print(f"Valid samples: {labels.count(0)}")
    print(f"Invalid samples: {labels.count(1)}")
    print(f"Train set: {len(X_train)} samples")
    print(f"Test set: {len(X_test)} samples")
    print(f"Train valid: {y_train.count(0)}, invalid: {y_train.count(1)}")
    print(f"Test valid: {y_test.count(0)}, invalid: {y_test.count(1)}\n")
    
    # Train model
    train_loader = DataLoader(X_train, batch_size=4, shuffle=True)
    test_loader = DataLoader(X_test, batch_size=4)
    
    model = SyntaxGCN(num_node_types=len(NODE_TYPE_TO_IDX))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = torch.nn.CrossEntropyLoss()
    
    # Training
    for epoch in range(30):
        model.train()
        total_loss = 0
        for batch in train_loader:
            optimizer.zero_grad()
            out = model(batch)
            loss = criterion(out, batch.y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
    
    # Detailed evaluation
    model.eval()
    all_predictions = []
    all_labels = []
    all_probabilities = []
    file_predictions = []
    
    with torch.no_grad():
        for batch in test_loader:
            out = model(batch)
            probabilities = torch.softmax(out, dim=1)
            pred = out.argmax(dim=1)
            
            all_predictions.extend(pred.cpu().numpy())
            all_labels.extend(batch.y.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())
    
    # Print detailed results
    print("=== DETAILED RESULTS ===")
    print(f"Test Accuracy: {np.mean(np.array(all_predictions) == np.array(all_labels)) * 100:.2f}%")
    print("\nClassification Report:")
    print(classification_report(all_labels, all_predictions, target_names=['Valid', 'Invalid']))
    
    print("\nConfusion Matrix:")
    cm = confusion_matrix(all_labels, all_predictions)
    print(cm)
    
    # Analyze confidence scores
    probabilities = np.array(all_probabilities)
    max_probs = np.max(probabilities, axis=1)
    print(f"\nConfidence Analysis:")
    print(f"Average confidence: {np.mean(max_probs):.4f}")
    print(f"Min confidence: {np.min(max_probs):.4f}")
    print(f"Max confidence: {np.max(max_probs):.4f}")
    
    # Check for overfitting indicators
    print(f"\nOverfitting Analysis:")
    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")
    
    # Analyze graph characteristics
    print(f"\nGraph Analysis:")
    train_graph_sizes = [g.x.size(0) for g in X_train]
    test_graph_sizes = [g.x.size(0) for g in X_test]
    print(f"Train graph sizes - Avg: {np.mean(train_graph_sizes):.1f}, Min: {min(train_graph_sizes)}, Max: {max(train_graph_sizes)}")
    print(f"Test graph sizes - Avg: {np.mean(test_graph_sizes):.1f}, Min: {min(test_graph_sizes)}, Max: {max(test_graph_sizes)}")
    
    return model, all_predictions, all_labels, test_files

def test_edge_cases():
    print("\n=== EDGE CASE TESTING ===")
    
    edge_cases = [
        ("empty_file", ""),
        ("single_line", "print('hello')"),
        ("complex_function", """
def complex_function(a, b, c):
    if a > b:
        return a + b
    elif b > c:
        return b * c
    else:
        return a + b + c
        """),
        ("nested_structures", """
class MyClass:
    def __init__(self):
        self.data = []
    
    def add_item(self, item):
        self.data.append(item)
        """),
        ("syntax_error_missing_colon", "def test()\n    pass"),
        ("syntax_error_indentation", "def test():\npass"),
        ("syntax_error_bracket", "print('hello'"),
        ("syntax_error_quote", "print('hello)"),
    ]
    
    model, _, _, _ = analyze_model_performance()
    
    for name, code in edge_cases:
        try:
            root = parse_code(code)
            if root is None:
                expected = 1  # Syntax error
            else:
                expected = 0  # Valid
                
            # Create graph
            if code.strip() == "":
                x = torch.tensor([[0, 0, 0]], dtype=torch.float)
                edge_index = torch.zeros((2, 0), dtype=torch.long)
                g = Data(x=x, edge_index=edge_index)
            else:
                g = ast_to_graph(code)
            
            g.batch = torch.zeros(g.x.size(0), dtype=torch.long)
            
            # Predict
            model.eval()
            with torch.no_grad():
                out = model(g)
                pred = out.argmax(dim=1).item()
                prob = torch.softmax(out, dim=1).max().item()
            
            print(f"{name:25} | Expected: {expected} | Predicted: {pred} | Confidence: {prob:.4f}")
            
        except Exception as e:
            print(f"{name:25} | Error: {str(e)}")

def check_data_leakage():
    print("\n=== DATA LEAKAGE CHECK ===")
    
    # Check if test files are too similar to train files
    samples, labels, filenames = load_dataset()
    
    print("File distribution:")
    valid_files = [f for f, l in zip(filenames, labels) if l == 0]
    invalid_files = [f for f, l in zip(filenames, labels) if l == 1]
    
    print(f"Valid files: {valid_files}")
    print(f"Invalid files: {invalid_files}")
    
    # Check for potential memorization
    print(f"\nDataset size: {len(samples)}")
    print(f"Unique code samples: {len(set(samples))}")
    
    # Check graph characteristics
    graphs = build_graphs(samples)
    graph_sizes = [g.x.size(0) for g in graphs]
    print(f"Graph size distribution: {graph_sizes}")

if __name__ == '__main__':
    analyze_model_performance()
    test_edge_cases()
    check_data_leakage() 