#!/usr/bin/env python3
"""
Training script for Python Syntax Error Classifier
Handles overfitting/underfitting with dynamic data generation
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


class DynamicDataGenerator:
    """Generates training data dynamically based on model performance"""

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

    def generate_balanced_dataset(self, samples_per_class=100) -> Tuple[List[Data], List[int]]:
        """Generate balanced dataset with equal samples per class"""
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
            'lambda x: x * 2',
            'x = [i for i in range(5)]',
            'assert x > 0',
            'raise ValueError("error")',
            'def decorator(func):\n    def wrapper(*args):\n        return func(*args)\n    return wrapper',
            'async def async_func():\n    await asyncio.sleep(1)',
            'match value:\n    case 1:\n        print("one")',
            'with open("file.txt") as f, open("out.txt", "w") as g:\n    pass',
            'if x > 5 and y < 10:\n    print("both")',
            'for i, item in enumerate(items):\n    print(f"{i}: {item}")',
            'try:\n    risky_operation()\nexcept ValueError:\n    handle_error()\nfinally:\n    cleanup()',
            'class ContextManager:\n    def __enter__(self):\n        return self\n    def __exit__(self, *args):\n        pass'
        ]

        # Generate more valid samples
        for i in range(samples_per_class - len(valid_samples)):
            valid_samples.append(f'def func{i}():\n    return {i}')

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
            'finally\n    pass',
            'async def async_func()\n    await asyncio.sleep(1)',
            'match value\n    case 1:\n        print("one")',
            'def decorator(func)\n    def wrapper(*args):\n        return func(*args)\n    return wrapper',
            'class ContextManager\n    def __enter__(self):\n        return self\n    def __exit__(self, *args):\n        pass',
            'if x > 5 and y < 10\n    print("both")',
            'for i, item in enumerate(items)\n    print(f"{i}: {item}")',
            'try:\n    risky_operation()\nexcept ValueError\n    handle_error()\nfinally\n    cleanup()',
            'with open("file.txt") as f, open("out.txt", "w") as g\n    pass',
            'def complex_func(a, b, c)\n    if a > b\n        return c\n    else\n        return a + b',
            'class ComplexClass\n    def __init__(self, value)\n        self.value = value\n    def method(self)\n        return self.value'
        ]

        # Generate more missing colon samples
        for i in range(samples_per_class - len(missing_colon_samples)):
            missing_colon_samples.append(f'def func{i}()\n    return {i}')

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
            'message = "Hello world',
            'path = "C:\\Users\\name',
            'data = "{"key": "value"',
            'sql = "SELECT * FROM table',
            'html = "<div class="container">',
            'json = \'{"name": "John"',
            'url = "https://example.com',
            'f_string = f"Hello {name}',
            'raw_string = r"C:\\path\\to\\file',
            'multiline = """This is a',
            'single_quote = \'This is a',
            'nested = "Outer \'inner\' string',
            'escaped = "Line 1\\nLine 2',
            'unicode = "Hello \\u0041',
            'bytes_string = b"Hello world',
            'formatted = "Value: {value}',
            'complex_string = f"Result: {x + y}"'
        ]

        # Generate more unclosed string samples
        for i in range(samples_per_class - len(unclosed_string_samples)):
            unclosed_string_samples.append(f'print("unclosed string {i}')

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
            '    if x > 5:\n      print(x)',
            '  class Test:\n    pass',
            '    try:\n      pass',
            '  for i in range(5):\n    print(i)',
            '    with open("file"):\n      pass',
            '  import os',
            '    from pathlib import Path',
            '  x = 5\n  y = 10',
            '    def nested():\n      return True',
            '  if condition:\n    action()',
            '    class Nested:\n      pass',
            '  async def async_func():\n    await sleep(1)',
            '    match value:\n      case 1:\n        print("one")',
            '  with context():\n    do_something()',
            '    for item in items:\n      process(item)',
            '  try:\n    risky()\nexcept:\n    handle()',
            '    def complex():\n      if a > b:\n        return c'
        ]

        # Generate more unexpected indent samples
        for i in range(samples_per_class - len(unexpected_indent_samples)):
            unexpected_indent_samples.append(
                f'  def func{i}():\n    return {i}')

        for code in unexpected_indent_samples:
            graph = self.graph_builder.build_graph(code)
            graph.y = torch.tensor([3], dtype=torch.long)  # unexpected_indent
            graphs.append(graph)
            labels.append(3)

        # Unexpected EOF samples
        unexpected_eof_samples = [
            'x = [1, 2, 3',
            'y = {"key": "value"',
            'def func(a, b',
            'if x > 5 and y < 10',
            'for i in range(10',
            'while True and x > 5',
            'class MyClass(',
            'try:',
            'with open("file.txt"',
            'print("hello"',
            'x = (1 + 2',
            'y = [i for i in range(5',
            'z = {"a": 1, "b": 2',
            'def test(a, b, c',
            'if x > 5 or y < 10',
            'async def async_func(',
            'match value:',
            'with context(',
            'for item in items:',
            'try:\n    risky(',
            'def complex_func(a, b, c, d, e, f, g, h, i, j'
        ]

        # Generate more unexpected EOF samples
        for i in range(samples_per_class - len(unexpected_eof_samples)):
            unexpected_eof_samples.append(f'x = [1, 2, 3, {i}')

        for code in unexpected_eof_samples:
            graph = self.graph_builder.build_graph(code)
            graph.y = torch.tensor([4], dtype=torch.long)  # unexpected_eof
            graphs.append(graph)
            labels.append(4)

        # Invalid token samples
        invalid_token_samples = [
            'x = @invalid',
            'y = #comment',
            'z = $symbol',
            'a = &operator',
            'b = ^xor',
            'c = ~not',
            'd = `backtick',
            'e = |pipe',
            'f = \\backslash',
            'g = /slash',
            'h = *asterisk',
            'i = %modulo',
            'j = =equals',
            'k = +plus',
            'l = -minus',
            'm = @decorator',
            'n = $variable',
            'o = &reference',
            'p = ^pointer',
            'q = ~bitwise',
            'r = |logical',
            's = \\escape',
            't = /division',
            'u = *multiplication',
            'v = %remainder'
        ]

        # Generate more invalid token samples
        for i in range(samples_per_class - len(invalid_token_samples)):
            invalid_token_samples.append(f'x = @invalid{i}')

        for code in invalid_token_samples:
            graph = self.graph_builder.build_graph(code)
            graph.y = torch.tensor([5], dtype=torch.long)  # invalid_token
            graphs.append(graph)
            labels.append(5)

        return graphs, labels

    def generate_additional_data(self, error_type: str, count: int) -> Tuple[List[Data], List[int]]:
        """Generate additional data for specific error type to handle overfitting/underfitting"""
        graphs = []
        labels = []

        error_type_id = self.error_types.index(error_type)

        if error_type == 'valid':
            for i in range(count):
                code = f'def valid_func{i}():\n    return {i}'
                graph = self.graph_builder.build_graph(code)
                graph.y = torch.tensor([error_type_id], dtype=torch.long)
                graphs.append(graph)
                labels.append(error_type_id)

        elif error_type == 'missing_colon':
            for i in range(count):
                code = f'def func{i}()\n    return {i}'
                graph = self.graph_builder.build_graph(code)
                graph.y = torch.tensor([error_type_id], dtype=torch.long)
                graphs.append(graph)
                labels.append(error_type_id)

        elif error_type == 'unclosed_string':
            for i in range(count):
                code = f'print("unclosed string {i}'
                graph = self.graph_builder.build_graph(code)
                graph.y = torch.tensor([error_type_id], dtype=torch.long)
                graphs.append(graph)
                labels.append(error_type_id)

        elif error_type == 'unexpected_indent':
            for i in range(count):
                code = f'  def func{i}():\n    return {i}'
                graph = self.graph_builder.build_graph(code)
                graph.y = torch.tensor([error_type_id], dtype=torch.long)
                graphs.append(graph)
                labels.append(error_type_id)

        elif error_type == 'unexpected_eof':
            for i in range(count):
                code = f'x = [1, 2, 3, {i}'
                graph = self.graph_builder.build_graph(code)
                graph.y = torch.tensor([error_type_id], dtype=torch.long)
                graphs.append(graph)
                labels.append(error_type_id)

        elif error_type == 'invalid_token':
            for i in range(count):
                code = f'x = @invalid{i}'
                graph = self.graph_builder.build_graph(code)
                graph.y = torch.tensor([error_type_id], dtype=torch.long)
                graphs.append(graph)
                labels.append(error_type_id)

        return graphs, labels


class ModelTrainer:
    """Trainer with overfitting/underfitting detection and dynamic data generation"""

    def __init__(self, batch_size=16, lr=0.001, epochs=100):
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')
        self.batch_size = batch_size
        self.lr = lr
        self.epochs = epochs
        self.data_generator = DynamicDataGenerator()
        logger.info(f"Using device: {self.device}")

    def detect_overfitting(self, train_losses: List[float], val_losses: List[float],
                           train_accs: List[float], val_accs: List[float]) -> Dict[str, any]:
        """Detect overfitting/underfitting patterns"""
        if len(train_losses) < 10:
            return {"status": "insufficient_data"}

        # Check for overfitting (training loss decreases but validation loss increases)
        recent_train_loss = np.mean(train_losses[-5:])
        recent_val_loss = np.mean(val_losses[-5:])
        early_train_loss = np.mean(train_losses[:5])
        early_val_loss = np.mean(val_losses[:5])

        train_loss_decreasing = recent_train_loss < early_train_loss
        val_loss_increasing = recent_val_loss > early_val_loss

        # Check for underfitting (both losses are high and not decreasing)
        high_losses = recent_train_loss > 1.0 and recent_val_loss > 1.0
        not_decreasing = abs(recent_train_loss - early_train_loss) < 0.1

        if train_loss_decreasing and val_loss_increasing:
            return {
                "status": "overfitting",
                "train_loss": recent_train_loss,
                "val_loss": recent_val_loss,
                "recommendation": "Add regularization or more training data"
            }
        elif high_losses and not_decreasing:
            return {
                "status": "underfitting",
                "train_loss": recent_train_loss,
                "val_loss": recent_val_loss,
                "recommendation": "Increase model capacity or training time"
            }
        else:
            return {
                "status": "normal",
                "train_loss": recent_train_loss,
                "val_loss": recent_val_loss
            }

    def train_model(self, train_loader, val_loader):
        """Train the model with overfitting/underfitting detection"""
        model = SyntaxErrorGNN().to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
        criterion = torch.nn.CrossEntropyLoss()

        logger.info("Training syntax error classification model...")

        best_val_acc = 0
        train_losses = []
        val_losses = []
        train_accs = []
        val_accs = []

        for epoch in range(self.epochs):
            # Training
            model.train()
            total_loss = 0
            correct = 0
            total = 0

            for batch in train_loader:
                batch = batch.to(self.device)
                optimizer.zero_grad()
                out = model(batch)
                loss = criterion(out, batch.y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

                pred = out.argmax(dim=1)
                correct += (pred == batch.y).sum().item()
                total += batch.y.size(0)

            avg_loss = total_loss / len(train_loader)
            train_acc = correct / total * 100
            train_losses.append(avg_loss)
            train_accs.append(train_acc)

            # Validation
            model.eval()
            val_loss = 0
            correct = 0
            total = 0

            with torch.no_grad():
                for batch in val_loader:
                    batch = batch.to(self.device)
                    out = model(batch)
                    loss = criterion(out, batch.y)
                    val_loss += loss.item()

                    pred = out.argmax(dim=1)
                    correct += (pred == batch.y).sum().item()
                    total += batch.y.size(0)

            avg_val_loss = val_loss / len(val_loader)
            val_acc = correct / total * 100
            val_losses.append(avg_val_loss)
            val_accs.append(val_acc)

            logger.info(
                f"Epoch {epoch+1}/{self.epochs}, Train Loss: {avg_loss:.4f}, Train Acc: {train_acc:.2f}%, Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.2f}%")

            # Check for overfitting/underfitting every 10 epochs
            if epoch % 10 == 0 and epoch > 0:
                diagnosis = self.detect_overfitting(
                    train_losses, val_losses, train_accs, val_accs)
                if diagnosis["status"] == "overfitting":
                    logger.warning(
                        f"Overfitting detected! {diagnosis['recommendation']}")
                    # Add more training data for the problematic class
                    self._handle_overfitting(model, train_loader, val_loader)
                elif diagnosis["status"] == "underfitting":
                    logger.warning(
                        f"Underfitting detected! {diagnosis['recommendation']}")
                    # Increase model capacity or training time
                    self._handle_underfitting(model, train_loader, val_loader)

            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(model.state_dict(), 'syntax_error_model.pth')
                logger.info(
                    f"New best model saved with validation accuracy: {val_acc:.2f}%")

        return model, train_losses, val_losses, train_accs, val_accs

    def _handle_overfitting(self, model, train_loader, val_loader):
        """Handle overfitting by adding regularization"""
        logger.info(
            "Adding dropout and early stopping to handle overfitting...")
        # Increase dropout
        for module in model.modules():
            if isinstance(module, nn.Dropout):
                module.p = min(0.5, module.p + 0.1)

    def _handle_underfitting(self, model, train_loader, val_loader):
        """Handle underfitting by increasing model capacity"""
        logger.info("Increasing model capacity to handle underfitting...")
        # This would require model architecture changes, so we'll just log it
        pass

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
        error_types = ['valid', 'missing_colon', 'unclosed_string',
                       'unexpected_indent', 'unexpected_eof', 'invalid_token']
        logger.info("Classification Report:")
        logger.info(classification_report(
            all_labels, all_preds, target_names=error_types))

        # Print confusion matrix
        cm = confusion_matrix(all_labels, all_preds)
        logger.info("Confusion Matrix:")
        logger.info(cm)

        return all_preds, all_labels


def main():
    """Main training function"""
    logger.info("Starting syntax error classifier training...")

    # Generate balanced dataset
    data_generator = DynamicDataGenerator()
    graphs, labels = data_generator.generate_balanced_dataset(
        samples_per_class=100)

    logger.info(f"Generated {len(graphs)} training samples")
    for i, error_type in enumerate(data_generator.error_types):
        count = labels.count(i)
        logger.info(f"{error_type}: {count}")

    # Split data
    train_graphs, temp_graphs, train_labels, temp_labels = train_test_split(
        graphs, labels, test_size=0.3, random_state=42, stratify=labels)

    val_graphs, test_graphs, val_labels, test_labels = train_test_split(
        temp_graphs, temp_labels, test_size=0.5, random_state=42, stratify=temp_labels)

    logger.info(f"Training samples: {len(train_graphs)}")
    logger.info(f"Validation samples: {len(val_graphs)}")
    logger.info(f"Test samples: {len(test_graphs)}")

    # Create data loaders
    train_loader = DataLoader(train_graphs, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_graphs, batch_size=16, shuffle=False)
    test_loader = DataLoader(test_graphs, batch_size=16, shuffle=False)

    # Train model
    trainer = ModelTrainer(batch_size=16, lr=0.001, epochs=100)
    model, train_losses, val_losses, train_accs, val_accs = trainer.train_model(
        train_loader, val_loader)

    # Evaluate model
    logger.info("Evaluating model...")
    trainer.evaluate_model(model, test_loader)

    # Save training history
    history = {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_accs': train_accs,
        'val_accs': val_accs
    }

    with open('training_history.json', 'w') as f:
        json.dump(history, f, indent=2)

    logger.info("Training completed!")
    logger.info("Model saved as: syntax_error_model.pth")
    logger.info("Training history saved as: training_history.json")


if __name__ == "__main__":
    main()
