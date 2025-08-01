#!/usr/bin/env python3
"""
Train the Syntax Error Detection Model
Trains a GCN model on the dataset and saves it for use in syntax_analyzer.py
"""

import os
import torch
import logging
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
from src.core.graph_builder import ast_to_graph, NODE_TYPE_TO_IDX, build_node_type_dict
from enhanced_model import EnhancedSyntaxGCN
from src.core.parser_util import parse_code
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
# Set up logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ModelTrainer:
    def __init__(self, data_dir='data', batch_size=8, lr=0.001, epochs=50):
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.lr = lr
        self.epochs = epochs
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")

    def load_dataset(self):
        """Load dataset from data directory"""
        samples = []
        labels = []
        filenames = []

        logger.info(f"Loading dataset from {self.data_dir}...")

        # Load valid samples
        valid_dir = os.path.join(self.data_dir, 'valid')
        if os.path.exists(valid_dir):
            for fname in os.listdir(valid_dir):
                if not fname.endswith('.py'):
                    continue
                try:
                    with open(os.path.join(valid_dir, fname), 'r', encoding='utf-8') as f:
                        code = f.read()
                    samples.append(code)
                    labels.append(0)  # Valid
                    filenames.append(f"valid/{fname}")
                except Exception as e:
                    logger.warning(f"Error reading {fname}: {e}")
                    continue

        # Load invalid samples
        invalid_dir = os.path.join(self.data_dir, 'invalid')
        if os.path.exists(invalid_dir):
            for fname in os.listdir(invalid_dir):
                if not fname.endswith('.py'):
                    continue
                try:
                    with open(os.path.join(invalid_dir, fname), 'r', encoding='utf-8') as f:
                        code = f.read()
                    samples.append(code)
                    labels.append(1)  # Invalid
                    filenames.append(f"invalid/{fname}")
                except Exception as e:
                    logger.warning(f"Error reading {fname}: {e}")
                    continue

        logger.info(
            f"Loaded {len(samples)} samples ({labels.count(0)} valid, {labels.count(1)} invalid)")
        return samples, labels, filenames

    def build_graphs(self, samples):
        """Build graphs from code samples"""
        graphs = []

        logger.info("Building graphs...")

        for i, code in enumerate(samples):
            try:
                g = ast_to_graph(code)
                if g is not None:
                    graphs.append(g)
                else:
                    # Create dummy graph for failed parsing
                    x = torch.tensor([[0, 0, 0]], dtype=torch.float)
                    edge_index = torch.zeros((2, 0), dtype=torch.long)
                    graphs.append(Data(x=x, edge_index=edge_index))
            except Exception as e:
                logger.warning(f"Error building graph for sample {i}: {e}")
                # Create dummy graph for failed parsing
                x = torch.tensor([[0, 0, 0]], dtype=torch.float)
                edge_index = torch.zeros((2, 0), dtype=torch.long)
                graphs.append(Data(x=x, edge_index=edge_index))

        return graphs

    def prepare_data(self):
        """Prepare data for training"""
        samples, labels, filenames = self.load_dataset()

        # Build node type dict for all valid samples
        logger.info("Building node type dictionary...")
        for code in samples:
            if code is not None:
                try:
                    root = parse_code(code)
                    if root is not None:
                        build_node_type_dict(root)
                except Exception as e:
                    logger.warning(f"Error building node type dict: {e}")

        graphs = self.build_graphs(samples)

        # Add labels and batch info
        for i, g in enumerate(graphs):
            g.y = torch.tensor([labels[i]], dtype=torch.long)
            g.batch = torch.zeros(g.x.size(0), dtype=torch.long)

        logger.info(f"Node types found: {len(NODE_TYPE_TO_IDX)}")
        logger.info(f"Graphs built: {len(graphs)}")

        return graphs, labels, filenames

    def train_model(self, train_loader, val_loader):
        """Train the model"""
        num_node_types = max(len(NODE_TYPE_TO_IDX), 50)
        model = EnhancedSyntaxGCN(
            num_node_types=num_node_types).to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
        criterion = torch.nn.CrossEntropyLoss()

        logger.info(f"Training model with {num_node_types} node types...")

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

            logger.info(
                f"Epoch {epoch+1}/{self.epochs}, Loss: {avg_loss:.4f}, Val Acc: {val_acc:.2f}%")

            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(model.state_dict(), 'syntax_error_model.pth')
                logger.info(
                    f"New best model saved with validation accuracy: {val_acc:.2f}%")

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
        logger.info("Classification Report:")
        logger.info(classification_report(
            all_labels, all_preds, target_names=['Valid', 'Invalid']))

        # Print confusion matrix
        cm = confusion_matrix(all_labels, all_preds)
        logger.info("Confusion Matrix:")
        logger.info(cm)

        return all_preds, all_labels

    def run_training(self):
        """Run the complete training pipeline"""
        logger.info("Starting model training...")

        # Prepare data
        graphs, labels, filenames = self.prepare_data()

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            graphs, labels, test_size=0.3, random_state=42, stratify=labels
        )

        # Create data loaders
        train_loader = DataLoader(
            X_train, batch_size=self.batch_size, shuffle=True)
        test_loader = DataLoader(X_test, batch_size=self.batch_size)

        logger.info(
            f"Training samples: {len(X_train)}, Test samples: {len(X_test)}")

        # Train model
        model, train_losses, val_accuracies = self.train_model(
            train_loader, test_loader)

        # Evaluate model
        logger.info("Evaluating model...")
        predictions, true_labels = self.evaluate_model(model, test_loader)

        # Save training history
        history = {
            'train_losses': train_losses,
            'val_accuracies': val_accuracies,
            'final_val_accuracy': val_accuracies[-1] if val_accuracies else 0,
            'node_types_count': len(NODE_TYPE_TO_IDX),
            'total_samples': len(graphs)
        }

        import json
        with open('training_history.json', 'w') as f:
            json.dump(history, f, indent=2)

        logger.info("Training completed!")
        logger.info(f"Model saved as: syntax_error_model.pth")
        logger.info(f"Training history saved as: training_history.json")

        return model


def main():
    """Main function"""
    trainer = ModelTrainer()

    try:
        model = trainer.run_training()
        print("\n✅ Training completed successfully!")
        print("Model saved as: syntax_error_model.pth")
        print("You can now use syntax_analyzer_gcn.py to analyze code!")

    except Exception as e:
        logger.error(f"Training failed: {e}")
        print(f"❌ Training failed: {e}")
        return 1

    return 0


if __name__ == '__main__':
    exit(main())
