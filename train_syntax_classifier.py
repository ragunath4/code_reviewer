#!/usr/bin/env python3
"""
Training script for the Python Syntax Error Classifier
Trains a GNN model to classify 6 types of syntax errors
"""

import torch
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
import numpy as np
import json
import logging
from pathlib import Path
from typing import List, Tuple, Dict
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import random

from syntax_error_classifier import TokenBasedGraphBuilder, SyntaxErrorGNN
from src.models.enhanced_model import UnifiedSyntaxGCN
from src.core.graph_builder import ast_to_graph

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SyntaxErrorDataset:
    """Dataset for syntax error classification"""
    def __init__(self, samples_per_class=100):
        self.error_types = [
            'valid',
            'missing_colon',
            'unclosed_string',
            'unexpected_indent',
            'unexpected_eof',
            'invalid_token'
        ]
        self.samples_per_class = samples_per_class

    def _expand_samples(self, base_samples, n):
        # Repeat and slightly modify samples to reach n
        expanded = []
        while len(expanded) < n:
            code = random.choice(base_samples)
            # Add a random comment or whitespace for variation
            if random.random() < 0.5:
                code += f"  # comment {random.randint(0, 1000)}"
            else:
                code = "\n" * random.randint(0, 2) + code
            expanded.append(code)
        return expanded[:n]

    def generate_training_data(self) -> Tuple[List[Data], List[Tuple[int, int]]]:
        """Generate training data with various syntax errors"""
        graphs = []
        labels = []
        # Valid code samples
        valid_samples = [
            'def test():\n    pass',
            'x = 5',
            'print("hello")',
            'if x > 5:\n    print(x)',
            'for i in range(10):\n    print(i)',
            'class MyClass:\n    def __init__(self):\n        pass',
            'try:\n    x = 1/0\nexcept:\n    pass',
            'with open("file.txt") as f:\n    pass',
            'def func(a, b):\n    return a + b',
            'x = [1, 2, 3]',
            'y = {"key": "value"}',
            'import os',
            'from pathlib import Path',
            'def decorator(func):\n    def wrapper(*args):\n        return func(*args)\n    return wrapper',
            'lambda x: x * 2',
            'x = [i for i in range(5)]',
            'x = {i: i*2 for i in range(3)}',
            'def gen():\n    for i in range(3):\n        yield i',
            'assert x > 0',
            'raise ValueError("error")'
        ]
        valid_samples = self._expand_samples(valid_samples, self.samples_per_class)
        for code in valid_samples:
            graph = ast_to_graph(code)
            if graph is None:
                continue
            graph.validity_label = torch.tensor([0], dtype=torch.long)  # valid
            graph.error_type_label = torch.tensor([0], dtype=torch.long)  # valid
            graphs.append(graph)
            labels.append((0, 0))
        # Error samples
        error_samples = [
            ([
                'def test()\n    pass',
                'if x > 5\n    print(x)',
                'for i in range(10)\n    print(i)',
                'while True\n    break',
                'class MyClass\n    pass',
                'try\n    pass\nexcept\n    pass',
                'with open("file.txt")\n    pass',
                'elif x > 5\n    print(x)',
                'else\n    print("else")',
                'finally\n    pass'
            ], 1),
            ([
                'print("hello',
                'x = "unclosed string',
                "text = 'missing quote",
                'print("hello\nworld"',
                'x = """unclosed triple quote',
                "y = 'single quote missing",
                'z = "double quote missing',
                'print("hello\\',
                'x = "nested "quotes" missing',
                "y = 'escaped \\' quote missing"
            ], 2),
            ([
                '  print("indented")',
                '    x = 5',
                '  def func():\n    pass',
                '  if x > 5:\n    print(x)',
                '  class Test:\n    pass',
                '  import os',
                '  x = 5\n  y = 10',
                '  try:\n    pass',
                '  for i in range(5):\n    print(i)',
                '  while True:\n    break'
            ], 3),
            ([
                'x = [1, 2, 3',
                'y = (1, 2, 3',
                'z = {1, 2, 3',
                'def func(a, b',
                'if x > 5 and',
                'for i in range(',
                'while x >',
                'class Test(',
                'try:',
                'with open('
            ], 4),
            ([
                'x = @invalid',
                'y = #invalid',
                'z = $invalid',
                'w = %invalid',
                'v = ^invalid',
                'u = &invalid',
                't = *invalid',
                's = +invalid',
                'r = -invalid',
                'q = =invalid'
            ], 5)
        ]
        for base_samples, error_type_id in error_samples:
            samples = self._expand_samples(base_samples, self.samples_per_class)
            for code in samples:
                graph = ast_to_graph(code)
                if graph is None:
                    continue
                graph.validity_label = torch.tensor([1], dtype=torch.long)  # invalid
                graph.error_type_label = torch.tensor([error_type_id], dtype=torch.long)
                graphs.append(graph)
                labels.append((1, error_type_id))
        logger.info(f"Generated {len(graphs)} training samples")
        for i, name in enumerate(self.error_types):
            count = sum(1 for v, e in labels if e == i)
            logger.info(f"{name}: {count}")
        return graphs, labels

class SyntaxErrorTrainer:
    """Trainer for syntax error classification"""
    
    def __init__(self, batch_size=16, lr=0.001, epochs=100):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.batch_size = batch_size
        self.lr = lr
        self.epochs = epochs
        logger.info(f"Using device: {self.device}")
    
    def train_model(self, train_loader, val_loader):
        """Train the syntax error classification model"""
        model = UnifiedSyntaxGCN(num_node_types=100, num_error_types=6).to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
        criterion_validity = torch.nn.CrossEntropyLoss()
        criterion_error_type = torch.nn.CrossEntropyLoss()
        logger.info("Training unified syntax error classification model...")
        best_val_acc = 0
        train_losses = []
        val_accuracies = []
        for epoch in range(self.epochs):
            # Training
            model.train()
            total_loss = 0
            for batch in train_loader:
                batch = batch.to(self.device)
                optimizer.zero_grad()
                validity_logits, error_type_logits = model(batch)
                loss_validity = criterion_validity(validity_logits, batch.validity_label.squeeze())
                loss_error_type = criterion_error_type(error_type_logits, batch.error_type_label.squeeze())
                loss = loss_validity + loss_error_type
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            avg_loss = total_loss / len(train_loader)
            train_losses.append(avg_loss)
            # Validation
            model.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for batch in val_loader:
                    batch = batch.to(self.device)
                    validity_logits, error_type_logits = model(batch)
                    pred_validity = validity_logits.argmax(dim=1)
                    correct += (pred_validity == batch.validity_label.squeeze()).sum().item()
                    total += batch.validity_label.size(0)
            val_acc = correct / total * 100
            val_accuracies.append(val_acc)
            logger.info(f"Epoch {epoch+1}/{self.epochs}, Loss: {avg_loss:.4f}, Val Acc: {val_acc:.2f}%")
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(model.state_dict(), 'unified_syntax_error_model.pth')
                logger.info(f"New best model saved with validation accuracy: {val_acc:.2f}%")
        return model, train_losses, val_accuracies
    
    def evaluate_model(self, model, test_loader):
        """Evaluate the trained model"""
        import numpy as np
        from sklearn.metrics import confusion_matrix
        model.eval()
        all_validity_preds = []
        all_error_type_preds = []
        all_validity_labels = []
        all_error_type_labels = []
        with torch.no_grad():
            for batch in test_loader:
                batch = batch.to(self.device)
                validity_logits, error_type_logits = model(batch)
                pred_validity = validity_logits.argmax(dim=1)
                pred_error_type = error_type_logits.argmax(dim=1)
                all_validity_preds.extend(pred_validity.cpu().numpy())
                all_error_type_preds.extend(pred_error_type.cpu().numpy())
                all_validity_labels.extend(batch.validity_label.squeeze().cpu().numpy())
                all_error_type_labels.extend(batch.error_type_label.squeeze().cpu().numpy())
        # Print classification report for validity
        logger.info("Validity Classification Report (valid/invalid):")
        logger.info(classification_report(all_validity_labels, all_validity_preds, target_names=["valid", "invalid"]))
        # Print confusion matrix for validity
        cm_validity = confusion_matrix(all_validity_labels, all_validity_preds)
        logger.info("Validity Confusion Matrix:")
        logger.info(f"\n{cm_validity}")
        # Print classification report for error type
        error_types = ['valid', 'missing_colon', 'unclosed_string', 'unexpected_indent', 'unexpected_eof', 'invalid_token']
        logger.info("Error Type Classification Report:")
        logger.info(classification_report(all_error_type_labels, all_error_type_preds, target_names=error_types))
        # Print confusion matrix for error type
        cm_error_type = confusion_matrix(all_error_type_labels, all_error_type_preds)
        logger.info("Error Type Confusion Matrix:")
        logger.info(f"\n{cm_error_type}")
        # Show a few example predictions
        logger.info("Sample predictions (true vs predicted):")
        for i in range(min(10, len(all_validity_labels))):
            logger.info(f"Sample {i}: Validity true={all_validity_labels[i]}, pred={all_validity_preds[i]}; ErrorType true={all_error_type_labels[i]}, pred={all_error_type_preds[i]}")
        return all_validity_preds, all_error_type_preds, all_validity_labels, all_error_type_labels

def main():
    """Main training function"""
    logger.info("Starting syntax error classifier training...")
    dataset = SyntaxErrorDataset(samples_per_class=100)
    graphs, labels = dataset.generate_training_data()
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        graphs, labels, test_size=0.2, random_state=42, stratify=labels
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
    )
    # Create data loaders
    train_loader = DataLoader(X_train, batch_size=16, shuffle=True)
    val_loader = DataLoader(X_val, batch_size=16, shuffle=False)
    test_loader = DataLoader(X_test, batch_size=16, shuffle=False)
    logger.info(f"Training samples: {len(X_train)}")
    logger.info(f"Validation samples: {len(X_val)}")
    logger.info(f"Test samples: {len(X_test)}")
    # Train model
    trainer = SyntaxErrorTrainer()
    model, train_losses, val_accuracies = trainer.train_model(train_loader, val_loader)
    # Evaluate model
    logger.info("Evaluating model...")
    # Unpack all four returned values
    validity_preds, error_type_preds, validity_labels, error_type_labels = trainer.evaluate_model(model, test_loader)
    # Save training history
    history = {
        'train_losses': train_losses,
        'val_accuracies': val_accuracies,
        'final_val_accuracy': val_accuracies[-1] if val_accuracies else 0,
        'total_samples': len(graphs)
    }
    with open('syntax_classifier_history.json', 'w') as f:
        json.dump(history, f, indent=2)
    logger.info("Training completed!")
    logger.info(f"Model saved as: unified_syntax_error_model.pth")
    logger.info(f"Training history saved as: syntax_classifier_history.json")
    return model

# Inference function for a single code sample

def ast_to_dict(node, depth=0, max_depth=10):
    if depth > max_depth:
        return {'type': node.type, 'children': '...'}
    d = {
        'type': node.type,
        'start_line': getattr(node, 'start_point', (0, 0))[0] + 1,
        'start_col': getattr(node, 'start_point', (0, 0))[1],
        'end_line': getattr(node, 'end_point', (0, 0))[0] + 1,
        'end_col': getattr(node, 'end_point', (0, 0))[1],
        'children': [ast_to_dict(child, depth+1, max_depth) for child in node.children]
    }
    return d

def analyze_code_with_details(model, code, device='cpu'):
    from src.core.graph_builder import ast_to_graph, NODE_TYPE_TO_IDX, build_node_type_dict
    from src.core.parser_util import parse_code
    import torch
    error_types = ['valid', 'missing_colon', 'unclosed_string', 'unexpected_indent', 'unexpected_eof', 'invalid_token']
    model.eval()
    # AST
    root = parse_code(code)
    if root is None:
        return {'validity': 'invalid', 'error_type': 'unknown', 'error_type_id': -1, 'error_nodes': [], 'ast': None, 'graph_features': None}
    ast_dict = ast_to_dict(root, max_depth=6)
    # Error nodes in AST
    error_nodes = []
    def find_errors(node):
        if node.type == 'ERROR':
            start_point = getattr(node, 'start_point', (0, 0))
            end_point = getattr(node, 'end_point', (0, 0))
            error_nodes.append({
                'start_line': start_point[0] + 1,
                'start_col': start_point[1],
                'end_line': end_point[0] + 1,
                'end_col': end_point[1],
            })
        for child in node.children:
            find_errors(child)
    find_errors(root)
    # Graph
    graph = ast_to_graph(code)
    if graph is None:
        return {'validity': 'invalid', 'error_type': 'unknown', 'error_type_id': -1, 'error_nodes': error_nodes, 'ast': ast_dict, 'graph_features': None}
    graph = graph.to(device)
    with torch.no_grad():
        validity_logits, error_type_logits = model(graph)
        pred_validity = validity_logits.argmax(dim=1).item()
        pred_error_type = error_type_logits.argmax(dim=1).item()
    validity_str = 'valid' if pred_validity == 0 else 'invalid'
    error_type_str = error_types[pred_error_type] if pred_error_type < len(error_types) else 'unknown'
    # Graph node features
    features = graph.x.cpu().numpy().tolist()
    graph_features = []
    for i, feat in enumerate(features):
        graph_features.append({
            'node_id': i,
            'node_type_idx': int(feat[0]),
            'depth': int(feat[1]),
            'num_children': int(feat[2]),
            'error_flag': int(feat[3]),
            'start_byte': int(feat[4]),
            'end_byte': int(feat[5]),
            'start_line': int(feat[6]),
            'start_col': int(feat[7]),
            'end_line': int(feat[8]),
            'end_col': int(feat[9]),
            'error_type_id': int(feat[10]),
        })
    return {
        'validity': validity_str,
        'error_type': error_type_str,
        'error_type_id': pred_error_type,
        'error_nodes': error_nodes,
        'ast': ast_dict,
        'graph_features': graph_features
    }

if __name__ == "__main__":
    main() 