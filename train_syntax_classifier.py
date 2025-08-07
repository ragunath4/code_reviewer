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

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SyntaxErrorDataset:
    """Dataset for syntax error classification"""
    
    def __init__(self):
        self.graph_builder = TokenBasedGraphBuilder()
        self.error_types = [
            'valid',
            'missing_colon',
            'unclosed_string', 
            'unexpected_indent',
            'unexpected_eof',
            'invalid_token'
        ]
    
    def generate_training_data(self) -> Tuple[List[Data], List[int]]:
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
        
        for code in valid_samples:
            graph = self.graph_builder.build_graph(code)
            graph.y = torch.tensor([0], dtype=torch.long)  # valid
            graphs.append(graph)
            labels.append(0)
        
        # Missing colon samples
        missing_colon_samples = [
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
        ]
        
        for code in missing_colon_samples:
            graph = self.graph_builder.build_graph(code)
            graph.y = torch.tensor([1], dtype=torch.long)  # missing_colon
            graphs.append(graph)
            labels.append(1)
        
        # Unclosed string samples
        unclosed_string_samples = [
            'print("hello',
            'x = "unclosed string',
            'text = \'missing quote',
            'print("hello\nworld"',
            'x = """unclosed triple quote',
            'y = \'single quote missing',
            'z = "double quote missing',
            'print("hello\\',
            'x = "nested "quotes" missing',
            'y = \'escaped \\\' quote missing'
        ]
        
        for code in unclosed_string_samples:
            graph = self.graph_builder.build_graph(code)
            graph.y = torch.tensor([2], dtype=torch.long)  # unclosed_string
            graphs.append(graph)
            labels.append(2)
        
        # Unexpected indent samples
        unexpected_indent_samples = [
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
        ]
        
        for code in unexpected_indent_samples:
            graph = self.graph_builder.build_graph(code)
            graph.y = torch.tensor([3], dtype=torch.long)  # unexpected_indent
            graphs.append(graph)
            labels.append(3)
        
        # Unexpected EOF samples
        unexpected_eof_samples = [
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
        ]
        
        for code in unexpected_eof_samples:
            graph = self.graph_builder.build_graph(code)
            graph.y = torch.tensor([4], dtype=torch.long)  # unexpected_eof
            graphs.append(graph)
            labels.append(4)
        
        # Invalid token samples
        invalid_token_samples = [
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
        ]
        
        for code in invalid_token_samples:
            graph = self.graph_builder.build_graph(code)
            graph.y = torch.tensor([5], dtype=torch.long)  # invalid_token
            graphs.append(graph)
            labels.append(5)
        
        logger.info(f"Generated {len(graphs)} training samples")
        logger.info(f"Valid: {labels.count(0)}")
        logger.info(f"Missing colon: {labels.count(1)}")
        logger.info(f"Unclosed string: {labels.count(2)}")
        logger.info(f"Unexpected indent: {labels.count(3)}")
        logger.info(f"Unexpected EOF: {labels.count(4)}")
        logger.info(f"Invalid token: {labels.count(5)}")
        
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
        model = SyntaxErrorGNN().to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
        criterion = torch.nn.CrossEntropyLoss()
        
        logger.info("Training syntax error classification model...")
        
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
                out = model(batch)
                loss = criterion(out, batch.y)
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
                    out = model(batch)
                    pred = out.argmax(dim=1)
                    correct += (pred == batch.y).sum().item()
                    total += batch.y.size(0)
            
            val_acc = correct / total * 100
            val_accuracies.append(val_acc)
            
            logger.info(f"Epoch {epoch+1}/{self.epochs}, Loss: {avg_loss:.4f}, Val Acc: {val_acc:.2f}%")
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(model.state_dict(), 'syntax_error_classifier_model.pth')
                logger.info(f"New best model saved with validation accuracy: {val_acc:.2f}%")
        
        return model, train_losses, val_accuracies
    
    def evaluate_model(self, model, test_loader):
        """Evaluate the trained model"""
        model.eval()
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in test_loader:
                batch = batch.to(self.device)
                out = model(batch)
                pred = out.argmax(dim=1)
                all_preds.extend(pred.cpu().numpy())
                all_labels.extend(batch.y.cpu().numpy())
        
        # Print classification report
        error_types = ['valid', 'missing_colon', 'unclosed_string', 'unexpected_indent', 'unexpected_eof', 'invalid_token']
        logger.info("Classification Report:")
        logger.info(classification_report(all_labels, all_preds, target_names=error_types))
        
        # Print confusion matrix
        cm = confusion_matrix(all_labels, all_preds)
        logger.info("Confusion Matrix:")
        logger.info(cm)
        
        return all_preds, all_labels

def main():
    """Main training function"""
    logger.info("Starting syntax error classifier training...")
    
    # Generate dataset
    dataset = SyntaxErrorDataset()
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
    predictions, true_labels = trainer.evaluate_model(model, test_loader)
    
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
    logger.info(f"Model saved as: syntax_error_classifier_model.pth")
    logger.info(f"Training history saved as: syntax_classifier_history.json")
    
    return model

if __name__ == "__main__":
    main() 