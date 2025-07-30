import os
import torch
import numpy as np
from torch_geometric.data import DataLoader, Data
from graph_builder import ast_to_graph, NODE_TYPE_TO_IDX
from model import SyntaxGCN
from parser_util import parse_code
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Tuple, Dict, Any
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ImprovedTrainer:
    def __init__(self, data_dir: str = 'data', batch_size: int = 8, lr: float = 0.001):
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.lr = lr
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
    def load_dataset(self) -> Tuple[List, List, List]:
        """Load dataset with proper error handling"""
        samples = []
        labels = []
        filenames = []
        
        for fname in os.listdir(self.data_dir):
            if not fname.endswith('.py'):
                continue
                
            try:
                with open(os.path.join(self.data_dir, fname), 'r', encoding='utf-8') as f:
                    code = f.read()
                
                # Parse code with error handling
                root = parse_code(code)
                if root is None:
                    samples.append(None)
                    labels.append(1)  # Syntax error
                else:
                    samples.append(code)
                    labels.append(0)  # Valid
                filenames.append(fname)
                
            except Exception as e:
                logger.warning(f"Error reading {fname}: {e}")
                continue
        
        logger.info(f"Loaded {len(samples)} samples ({labels.count(0)} valid, {labels.count(1)} invalid)")
        return samples, labels, filenames
    
    def build_graphs(self, samples: List) -> List[Data]:
        """Build graphs with proper error handling"""
        graphs = []
        
        for i, code in enumerate(samples):
            try:
                if code is None:
                    # Dummy graph for syntax error
                    x = torch.tensor([[0, 0, 0]], dtype=torch.float)
                    edge_index = torch.zeros((2, 0), dtype=torch.long)
                    graphs.append(Data(x=x, edge_index=edge_index))
                else:
                    g = ast_to_graph(code)
                    graphs.append(g)
            except Exception as e:
                logger.warning(f"Error building graph for sample {i}: {e}")
                # Create dummy graph for failed parsing
                x = torch.tensor([[0, 0, 0]], dtype=torch.float)
                edge_index = torch.zeros((2, 0), dtype=torch.long)
                graphs.append(Data(x=x, edge_index=edge_index))
        
        return graphs
    
    def prepare_data(self) -> Tuple[List[Data], List[int], List[str]]:
        """Prepare data with node type dictionary building"""
        samples, labels, filenames = self.load_dataset()
        
        # Build node type dict for all valid samples
        for code in samples:
            if code is not None:
                try:
                    root = parse_code(code)
                    if root is not None:
                        from graph_builder import build_node_type_dict
                        build_node_type_dict(root)
                except Exception as e:
                    logger.warning(f"Error building node type dict: {e}")
        
        graphs = self.build_graphs(samples)
        
        # Add labels and batch info
        for i, g in enumerate(graphs):
            g.y = torch.tensor([labels[i]], dtype=torch.long)
            g.batch = torch.zeros(g.x.size(0), dtype=torch.long)
        
        return graphs, labels, filenames
    
    def cross_validation(self, n_folds: int = 5) -> Dict[str, List[float]]:
        """Perform cross-validation"""
        graphs, labels, filenames = self.prepare_data()
        
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
        fold_results = {
            'train_acc': [], 'val_acc': [], 'test_acc': [],
            'train_loss': [], 'val_loss': [], 'test_loss': []
        }
        
        for fold, (train_val_idx, test_idx) in enumerate(skf.split(graphs, labels)):
            logger.info(f"Training fold {fold + 1}/{n_folds}")
            
            # Split train/val
            train_idx, val_idx = train_test_split(
                train_val_idx, test_size=0.2, random_state=42, 
                stratify=[labels[i] for i in train_val_idx]
            )
            
            # Create data loaders
            train_graphs = [graphs[i] for i in train_idx]
            val_graphs = [graphs[i] for i in val_idx]
            test_graphs = [graphs[i] for i in test_idx]
            
            train_loader = DataLoader(train_graphs, batch_size=self.batch_size, shuffle=True)
            val_loader = DataLoader(val_graphs, batch_size=self.batch_size)
            test_loader = DataLoader(test_graphs, batch_size=self.batch_size)
            
            # Train model
            model = SyntaxGCN(num_node_types=len(NODE_TYPE_TO_IDX)).to(self.device)
            optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
            criterion = torch.nn.CrossEntropyLoss()
            
            best_val_acc = 0
            patience = 10
            patience_counter = 0
            
            for epoch in range(100):  # Increased epochs
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
                
                if (epoch + 1) % 10 == 0:
                    logger.info(f"Epoch {epoch + 1}: Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}")
            
            # Test evaluation
            model.eval()
            test_loss = 0
            test_correct = 0
            test_total = 0
            
            with torch.no_grad():
                for batch in test_loader:
                    batch = batch.to(self.device)
                    out = model(batch)
                    loss = criterion(out, batch.y)
                    test_loss += loss.item()
                    pred = out.argmax(dim=1)
                    test_correct += (pred == batch.y).sum().item()
                    test_total += batch.y.size(0)
            
            test_acc = test_correct / test_total
            avg_test_loss = test_loss / len(test_loader)
            
            # Store results
            fold_results['train_acc'].append(train_acc)
            fold_results['val_acc'].append(val_acc)
            fold_results['test_acc'].append(test_acc)
            fold_results['train_loss'].append(avg_train_loss)
            fold_results['val_loss'].append(avg_val_loss)
            fold_results['test_loss'].append(avg_test_loss)
            
            logger.info(f"Fold {fold + 1} Results:")
            logger.info(f"  Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}, Test Acc: {test_acc:.4f}")
        
        return fold_results
    
    def train_final_model(self) -> Tuple[SyntaxGCN, Dict[str, float]]:
        """Train final model on full dataset"""
        graphs, labels, filenames = self.prepare_data()
        
        # Split into train/val/test
        X_train, X_temp, y_train, y_temp = train_test_split(
            graphs, labels, test_size=0.3, random_state=42, stratify=labels
        )
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=0.5, random_state=42, 
            stratify=y_temp
        )
        
        train_loader = DataLoader(X_train, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(X_val, batch_size=self.batch_size)
        test_loader = DataLoader(X_test, batch_size=self.batch_size)
        
        model = SyntaxGCN(num_node_types=len(NODE_TYPE_TO_IDX)).to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
        criterion = torch.nn.CrossEntropyLoss()
        
        best_val_acc = 0
        patience = 15
        patience_counter = 0
        
        for epoch in range(150):
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
            
            if (epoch + 1) % 10 == 0:
                logger.info(f"Epoch {epoch + 1}: Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}")
        
        # Final evaluation
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
        
        results = {
            'train_acc': train_acc,
            'val_acc': val_acc,
            'test_acc': test_acc,
            'train_loss': avg_train_loss,
            'val_loss': avg_val_loss,
            'test_loss': avg_test_loss
        }
        
        logger.info("Final Model Results:")
        logger.info(f"  Train Acc: {train_acc:.4f}")
        logger.info(f"  Val Acc: {val_acc:.4f}")
        logger.info(f"  Test Acc: {test_acc:.4f}")
        
        # Print detailed metrics
        print("\nClassification Report:")
        print(classification_report(all_labels, all_predictions, target_names=['Valid', 'Invalid']))
        
        print("\nConfusion Matrix:")
        cm = confusion_matrix(all_labels, all_predictions)
        print(cm)
        
        return model, results
    
    def test_generalization(self, model: SyntaxGCN) -> float:
        """Test model on new, unseen cases"""
        test_cases = [
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
            ("syntax_error_missing_colon", "def test()\n    pass"),
            ("syntax_error_bracket", "print('hello'"),
            ("syntax_error_quote", "print('hello)"),
            ("nested_structures", """
class MyClass:
    def __init__(self):
        self.data = []
    
    def add_item(self, item):
        self.data.append(item)
            """),
        ]
        
        model.eval()
        correct = 0
        total = 0
        
        for name, code in test_cases:
            try:
                root = parse_code(code)
                expected = 1 if root is None else 0
                
                if code.strip() == "":
                    x = torch.tensor([[0, 0, 0]], dtype=torch.float)
                    edge_index = torch.zeros((2, 0), dtype=torch.long)
                    g = Data(x=x, edge_index=edge_index)
                else:
                    g = ast_to_graph(code)
                
                g.batch = torch.zeros(g.x.size(0), dtype=torch.long)
                g = g.to(self.device)
                
                with torch.no_grad():
                    out = model(g)
                    pred = out.argmax(dim=1).item()
                
                if pred == expected:
                    correct += 1
                total += 1
                
            except Exception as e:
                logger.warning(f"Error testing {name}: {e}")
                total += 1
        
        generalization_acc = correct / total if total > 0 else 0
        logger.info(f"Generalization Accuracy: {generalization_acc:.4f} ({correct}/{total})")
        return generalization_acc

def main():
    # Create expanded dataset
    logger.info("Creating expanded dataset...")
    from expanded_dataset import create_expanded_dataset
    create_expanded_dataset()
    
    # Initialize trainer
    trainer = ImprovedTrainer()
    
    # Cross-validation
    logger.info("Running cross-validation...")
    cv_results = trainer.cross_validation(n_folds=5)
    
    # Print CV results
    print("\nCross-Validation Results:")
    for metric in ['train_acc', 'val_acc', 'test_acc']:
        values = cv_results[metric]
        print(f"{metric}: {np.mean(values):.4f} ± {np.std(values):.4f}")
    
    # Train final model
    logger.info("Training final model...")
    model, results = trainer.train_final_model()
    
    # Test generalization
    logger.info("Testing generalization...")
    gen_acc = trainer.test_generalization(model)
    
    # Save model
    torch.save(model.state_dict(), 'syntax_error_model.pth')
    logger.info("Model saved to syntax_error_model.pth")

if __name__ == '__main__':
    main() 