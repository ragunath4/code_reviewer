import os
import torch
import numpy as np
from torch_geometric.data import DataLoader
from graph_builder import ast_to_graph, NODE_TYPE_TO_IDX
from enhanced_model import SyntaxGCN, EnhancedSyntaxGCN, AttentionSyntaxGCN, ResidualSyntaxGCN
from parser_util import parse_code
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelComparator:
    def __init__(self, data_dir: str = 'data', batch_size: int = 8, lr: float = 0.001):
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.lr = lr
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def load_and_prepare_data(self) -> Tuple[List, List, List]:
        """Load and prepare data"""
        samples = []
        labels = []
        filenames = []
        
        for fname in os.listdir(self.data_dir):
            if not fname.endswith('.py'):
                continue
            try:
                with open(os.path.join(self.data_dir, fname), 'r', encoding='utf-8') as f:
                    code = f.read()
                root = parse_code(code)
                if root is None:
                    samples.append(None)
                    labels.append(1)
                else:
                    samples.append(code)
                    labels.append(0)
                filenames.append(fname)
            except Exception as e:
                logger.warning(f"Error reading {fname}: {e}")
                continue
        
        # Build node type dict
        for code in samples:
            if code is not None:
                try:
                    root = parse_code(code)
                    if root is not None:
                        from graph_builder import build_node_type_dict
                        build_node_type_dict(root)
                except Exception as e:
                    logger.warning(f"Error building node type dict: {e}")
        
        # Build graphs
        graphs = []
        for i, code in enumerate(samples):
            try:
                if code is None:
                    x = torch.tensor([[0, 0, 0]], dtype=torch.float)
                    edge_index = torch.zeros((2, 0), dtype=torch.long)
                    graphs.append(Data(x=x, edge_index=edge_index))
                else:
                    g = ast_to_graph(code)
                    graphs.append(g)
            except Exception as e:
                logger.warning(f"Error building graph for sample {i}: {e}")
                x = torch.tensor([[0, 0, 0]], dtype=torch.float)
                edge_index = torch.zeros((2, 0), dtype=torch.long)
                graphs.append(Data(x=x, edge_index=edge_index))
        
        # Add labels and batch info
        for i, g in enumerate(graphs):
            g.y = torch.tensor([labels[i]], dtype=torch.long)
            g.batch = torch.zeros(g.x.size(0), dtype=torch.long)
        
        return graphs, labels, filenames
    
    def train_model(self, model_class, model_name: str, graphs: List, labels: List) -> Dict:
        """Train a single model and return results"""
        logger.info(f"Training {model_name}...")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            graphs, labels, test_size=0.3, random_state=42, stratify=labels
        )
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
        )
        
        # Create data loaders
        train_loader = DataLoader(X_train, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(X_val, batch_size=self.batch_size)
        test_loader = DataLoader(X_test, batch_size=self.batch_size)
        
        # Initialize model
        if model_name == "EnhancedSyntaxGCN":
            model = model_class(num_node_types=len(NODE_TYPE_TO_IDX), hidden_dim=64, dropout=0.3)
        elif model_name == "AttentionSyntaxGCN":
            model = model_class(num_node_types=len(NODE_TYPE_TO_IDX), hidden_dim=64, dropout=0.3)
        elif model_name == "ResidualSyntaxGCN":
            model = model_class(num_node_types=len(NODE_TYPE_TO_IDX), hidden_dim=64, dropout=0.3)
        else:
            model = model_class(num_node_types=len(NODE_TYPE_TO_IDX))
        
        model = model.to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
        criterion = torch.nn.CrossEntropyLoss()
        
        # Training loop
        best_val_acc = 0
        patience = 15
        patience_counter = 0
        
        for epoch in range(100):
            # Training
            model.train()
            train_loss = 0
            train_correct = 0
            train_total = 0
            
            for batch in train_loader:
                batch = batch.to(self.device)
                optimizer.zero_grad()
                out = model(batch)
                loss = criterion(out, batch.y)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                pred = out.argmax(dim=1)
                train_correct += (pred == batch.y).sum().item()
                train_total += batch.y.size(0)
            
            # Validation
            model.eval()
            val_loss = 0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for batch in val_loader:
                    batch = batch.to(self.device)
                    out = model(batch)
                    loss = criterion(out, batch.y)
                    val_loss += loss.item()
                    pred = out.argmax(dim=1)
                    val_correct += (pred == batch.y).sum().item()
                    val_total += batch.y.size(0)
            
            train_acc = train_correct / train_total
            val_acc = val_correct / val_total
            avg_train_loss = train_loss / len(train_loader)
            avg_val_loss = val_loss / len(val_loader)
            
            # Early stopping
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
            else:
                patience_counter += 1
            
            if patience_counter >= patience:
                logger.info(f"Early stopping at epoch {epoch + 1}")
                break
        
        # Test evaluation
        model.eval()
        test_loss = 0
        test_correct = 0
        test_total = 0
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for batch in test_loader:
                batch = batch.to(self.device)
                out = model(batch)
                loss = criterion(out, batch.y)
                test_loss += loss.item()
                pred = out.argmax(dim=1)
                test_correct += (pred == batch.y).sum().item()
                test_total += batch.y.size(0)
                
                all_predictions.extend(pred.cpu().numpy())
                all_labels.extend(batch.y.cpu().numpy())
        
        test_acc = test_correct / test_total
        avg_test_loss = test_loss / len(test_loader)
        
        # Calculate additional metrics
        from sklearn.metrics import precision_recall_fscore_support
        precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_predictions, average='weighted')
        
        results = {
            'model_name': model_name,
            'train_acc': train_acc,
            'val_acc': val_acc,
            'test_acc': test_acc,
            'train_loss': avg_train_loss,
            'val_loss': avg_val_loss,
            'test_loss': avg_test_loss,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'num_params': sum(p.numel() for p in model.parameters())
        }
        
        logger.info(f"{model_name} Results:")
        logger.info(f"  Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}, Test Acc: {test_acc:.4f}")
        logger.info(f"  Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
        logger.info(f"  Parameters: {results['num_params']:,}")
        
        return results
    
    def compare_models(self) -> Dict[str, Dict]:
        """Compare all model architectures"""
        graphs, labels, filenames = self.load_and_prepare_data()
        
        models = {
            "OriginalSyntaxGCN": SyntaxGCN,
            "EnhancedSyntaxGCN": EnhancedSyntaxGCN,
            "AttentionSyntaxGCN": AttentionSyntaxGCN,
            "ResidualSyntaxGCN": ResidualSyntaxGCN
        }
        
        results = {}
        
        for model_name, model_class in models.items():
            try:
                result = self.train_model(model_class, model_name, graphs, labels)
                results[model_name] = result
            except Exception as e:
                logger.error(f"Error training {model_name}: {e}")
                results[model_name] = {'error': str(e)}
        
        return results
    
    def plot_comparison(self, results: Dict[str, Dict]):
        """Plot comparison of model performances"""
        # Filter out models with errors
        valid_results = {k: v for k, v in results.items() if 'error' not in v}
        
        if not valid_results:
            logger.error("No valid results to plot")
            return
        
        # Create comparison plots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Accuracy comparison
        model_names = list(valid_results.keys())
        train_accs = [valid_results[name]['train_acc'] for name in model_names]
        val_accs = [valid_results[name]['val_acc'] for name in model_names]
        test_accs = [valid_results[name]['test_acc'] for name in model_names]
        
        x = np.arange(len(model_names))
        width = 0.25
        
        axes[0, 0].bar(x - width, train_accs, width, label='Train', alpha=0.8)
        axes[0, 0].bar(x, val_accs, width, label='Validation', alpha=0.8)
        axes[0, 0].bar(x + width, test_accs, width, label='Test', alpha=0.8)
        axes[0, 0].set_xlabel('Model')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].set_title('Model Accuracy Comparison')
        axes[0, 0].set_xticks(x)
        axes[0, 0].set_xticklabels(model_names, rotation=45)
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Loss comparison
        train_losses = [valid_results[name]['train_loss'] for name in model_names]
        val_losses = [valid_results[name]['val_loss'] for name in model_names]
        test_losses = [valid_results[name]['test_loss'] for name in model_names]
        
        axes[0, 1].bar(x - width, train_losses, width, label='Train', alpha=0.8)
        axes[0, 1].bar(x, val_losses, width, label='Validation', alpha=0.8)
        axes[0, 1].bar(x + width, test_losses, width, label='Test', alpha=0.8)
        axes[0, 1].set_xlabel('Model')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].set_title('Model Loss Comparison')
        axes[0, 1].set_xticks(x)
        axes[0, 1].set_xticklabels(model_names, rotation=45)
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # F1 Score comparison
        f1_scores = [valid_results[name]['f1'] for name in model_names]
        precision_scores = [valid_results[name]['precision'] for name in model_names]
        recall_scores = [valid_results[name]['recall'] for name in model_names]
        
        axes[1, 0].bar(x - width, precision_scores, width, label='Precision', alpha=0.8)
        axes[1, 0].bar(x, recall_scores, width, label='Recall', alpha=0.8)
        axes[1, 0].bar(x + width, f1_scores, width, label='F1', alpha=0.8)
        axes[1, 0].set_xlabel('Model')
        axes[1, 0].set_ylabel('Score')
        axes[1, 0].set_title('Model Metrics Comparison')
        axes[1, 0].set_xticks(x)
        axes[1, 0].set_xticklabels(model_names, rotation=45)
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Parameter count comparison
        param_counts = [valid_results[name]['num_params'] for name in model_names]
        
        axes[1, 1].bar(x, param_counts, alpha=0.8)
        axes[1, 1].set_xlabel('Model')
        axes[1, 1].set_ylabel('Number of Parameters')
        axes[1, 1].set_title('Model Complexity Comparison')
        axes[1, 1].set_xticks(x)
        axes[1, 1].set_xticklabels(model_names, rotation=45)
        axes[1, 1].grid(True, alpha=0.3)
        
        # Add value labels on bars
        for ax in axes.flat:
            for container in ax.containers:
                ax.bar_label(container, fmt='%.3f')
        
        plt.tight_layout()
        plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Print summary table
        print("\n" + "="*80)
        print("MODEL COMPARISON SUMMARY")
        print("="*80)
        print(f"{'Model':<20} {'Test Acc':<10} {'F1':<10} {'Params':<10} {'Train Acc':<10} {'Val Acc':<10}")
        print("-"*80)
        
        for model_name in model_names:
            result = valid_results[model_name]
            print(f"{model_name:<20} {result['test_acc']:<10.4f} {result['f1']:<10.4f} "
                  f"{result['num_params']:<10,} {result['train_acc']:<10.4f} {result['val_acc']:<10.4f}")
        
        print("="*80)
        
        # Find best model
        best_model = max(valid_results.keys(), key=lambda x: valid_results[x]['test_acc'])
        print(f"\nBest Model: {best_model}")
        print(f"Test Accuracy: {valid_results[best_model]['test_acc']:.4f}")
        print(f"F1 Score: {valid_results[best_model]['f1']:.4f}")

def main():
    # Create expanded dataset if it doesn't exist
    if not os.path.exists('data/valid_20.py'):
        logger.info("Creating expanded dataset...")
        from expanded_dataset import create_expanded_dataset
        create_expanded_dataset()
    
    # Run model comparison
    comparator = ModelComparator()
    results = comparator.compare_models()
    
    # Plot results
    comparator.plot_comparison(results)
    
    # Save results
    import json
    with open('model_comparison_results.json', 'w') as f:
        # Convert numpy types to native Python types for JSON serialization
        json_results = {}
        for model_name, result in results.items():
            if 'error' not in result:
                json_results[model_name] = {
                    k: float(v) if isinstance(v, (np.floating, float)) else v
                    for k, v in result.items()
                }
            else:
                json_results[model_name] = result
        json.dump(json_results, f, indent=2)
    
    logger.info("Results saved to model_comparison_results.json")

if __name__ == '__main__':
    main() 