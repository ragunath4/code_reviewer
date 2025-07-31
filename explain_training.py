#!/usr/bin/env python3
"""
Explanation of Model Training Process
This script demonstrates how the model gets trained using data from the data/ folder
and how Python files are converted to graphs for training.
"""

import os
import torch
from torch_geometric.data import Data
from graph_builder import ast_to_graph, NODE_TYPE_TO_IDX, build_node_type_dict
from parser_util import parse_code
from enhanced_model import EnhancedSyntaxGCN


def explain_data_structure():
    """Explain the data folder structure and how files are organized"""
    print("📁 DATA FOLDER STRUCTURE")
    print("=" * 50)

    data_dir = "data"
    valid_files = [f for f in os.listdir(data_dir) if f.startswith("valid_")]
    invalid_files = [f for f in os.listdir(
        data_dir) if f.startswith("invalid_")]

    print(f"✅ Valid Python files: {len(valid_files)}")
    print(f"❌ Invalid Python files: {len(invalid_files)}")
    print(f"📊 Total training samples: {len(valid_files) + len(invalid_files)}")

    print("\n📋 Sample Files:")
    print("Valid files (should parse successfully):")
    for i, f in enumerate(valid_files[:5]):
        print(f"  - {f}")
    print("  ...")

    print("\nInvalid files (should have syntax errors):")
    for i, f in enumerate(invalid_files[:5]):
        print(f"  - {f}")
    print("  ...")

    return valid_files, invalid_files


def demonstrate_file_loading():
    """Show how individual files are loaded and processed"""
    print("\n🔄 FILE LOADING PROCESS")
    print("=" * 50)

    # Example 1: Valid file
    print("\n1️⃣ Loading Valid File (valid_01.py):")
    with open("data/valid_01.py", "r") as f:
        code = f.read()
    print(f"Code: {code.strip()}")

    # Parse the code
    root = parse_code(code)
    if root:
        print("✅ Parser: Code is valid (no syntax errors)")
    else:
        print("❌ Parser: Code has syntax errors")

    # Example 2: Invalid file
    print("\n2️⃣ Loading Invalid File (invalid_01.py):")
    with open("data/invalid_01.py", "r") as f:
        code = f.read()
    print(f"Code: {code.strip()}")

    # Parse the code
    root = parse_code(code)
    if root:
        print("✅ Parser: Code is valid (no syntax errors)")
    else:
        print("❌ Parser: Code has syntax errors")

    return code


def demonstrate_graph_conversion():
    """Show how code is converted to graphs"""
    print("\n🔄 GRAPH CONVERSION PROCESS")
    print("=" * 50)

    # Example with valid code
    valid_code = """
def greet(name):
    print("Hello", name)
    return name
"""
    print("\n1️⃣ Converting Valid Code to Graph:")
    print(f"Code: {valid_code.strip()}")

    # Build node type dictionary
    root = parse_code(valid_code)
    if root:
        build_node_type_dict(root)
        print(f"✅ Node types found: {len(NODE_TYPE_TO_IDX)}")

        # Convert to graph
        graph = ast_to_graph(valid_code)
        if graph:
            print(f"✅ Graph created successfully!")
            print(f"   - Nodes: {graph.x.shape[0]}")
            print(f"   - Features per node: {graph.x.shape[1]}")
            print(f"   - Edges: {graph.edge_index.shape[1]}")
            print(f"   - Node features shape: {graph.x.shape}")
            print(f"   - Edge index shape: {graph.edge_index.shape}")
        else:
            print("❌ Failed to create graph")
    else:
        print("❌ Code has syntax errors")

    # Example with invalid code
    invalid_code = """
def greet(name)
    print("Hello", name)
"""
    print("\n2️⃣ Converting Invalid Code to Graph:")
    print(f"Code: {invalid_code.strip()}")

    graph = ast_to_graph(invalid_code)
    if graph is None:
        print("✅ Graph builder correctly returned None for invalid code")
        print("   This prevents the GNN from processing invalid syntax")
    else:
        print("❌ Graph builder should have returned None")


def demonstrate_training_data_preparation():
    """Show how training data is prepared"""
    print("\n🔄 TRAINING DATA PREPARATION")
    print("=" * 50)

    # Simulate the training data preparation process
    samples = []
    labels = []

    # Load a few sample files
    sample_files = [
        ("data/valid_01.py", 0),  # 0 = valid
        ("data/invalid_01.py", 1),  # 1 = invalid
        ("data/valid_12.py", 0),
        ("data/invalid_12.py", 1)
    ]

    print("📂 Loading files and creating labels:")
    for filepath, label in sample_files:
        try:
            with open(filepath, 'r') as f:
                code = f.read()

            # Check if code is valid
            root = parse_code(code)
            actual_label = 1 if root is None else 0

            print(f"  {filepath}: Label={label}, Actual={actual_label}")
            samples.append(code)
            labels.append(actual_label)

        except Exception as e:
            print(f"  Error reading {filepath}: {e}")

    print(f"\n📊 Dataset Summary:")
    print(f"  Total samples: {len(samples)}")
    print(f"  Valid samples: {labels.count(0)}")
    print(f"  Invalid samples: {labels.count(1)}")

    return samples, labels


def demonstrate_model_training():
    """Show how the model is trained on graphs"""
    print("\n🔄 MODEL TRAINING PROCESS")
    print("=" * 50)

    # Create sample graphs
    valid_code = "def test(): return 42"
    invalid_code = "def test() return 42"  # Missing colon

    print("1️⃣ Creating sample graphs for training:")

    # Build node type dict
    root = parse_code(valid_code)
    if root:
        build_node_type_dict(root)

    # Create graphs
    valid_graph = ast_to_graph(valid_code)
    invalid_graph = ast_to_graph(invalid_code)

    if valid_graph:
        valid_graph.y = torch.tensor([0])  # Valid label
        valid_graph.batch = torch.zeros(
            valid_graph.x.size(0), dtype=torch.long)
        print(f"✅ Valid graph: {valid_graph.x.shape[0]} nodes, label=0")

    if invalid_graph is None:
        # Create dummy graph for invalid code
        x = torch.tensor([[0, 0, 0]], dtype=torch.float)
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        invalid_graph = Data(x=x, edge_index=edge_index)
        invalid_graph.y = torch.tensor([1])  # Invalid label
        invalid_graph.batch = torch.zeros(1, dtype=torch.long)
        print("✅ Invalid graph: dummy graph created, label=1")

    # Initialize model
    num_node_types = max(len(NODE_TYPE_TO_IDX), 50)
    model = EnhancedSyntaxGCN(num_node_types=num_node_types)
    print(f"\n2️⃣ Model initialized:")
    print(f"   - Node types: {num_node_types}")
    print(f"   - Model type: {type(model).__name__}")

    # Demonstrate forward pass
    print("\n3️⃣ Forward pass demonstration:")
    if valid_graph:
        # Set model to evaluation mode to avoid BatchNorm issues
        model.eval()
        with torch.no_grad():
            try:
                output = model(valid_graph)
                probabilities = torch.softmax(output, dim=1)
                prediction = torch.argmax(probabilities, dim=1)
                confidence = torch.max(probabilities, dim=1)[0]

                print(
                    f"   Valid code prediction: {'Valid' if prediction.item() == 0 else 'Invalid'}")
                print(f"   Confidence: {confidence.item():.3f}")
            except Exception as e:
                print(f"   ⚠️ Forward pass failed: {e}")
                print(f"   This is expected due to BatchNorm requiring multiple samples")
                print(f"   In real training, graphs are batched together")

    print("\n4️⃣ Training Process Summary:")
    print("   - Files are loaded from data/ folder")
    print("   - Each file is parsed using Tree-sitter")
    print("   - Valid code → AST → Graph")
    print("   - Invalid code → Dummy graph")
    print("   - Graphs are batched and fed to GNN")
    print("   - Model learns to classify graphs as valid/invalid")

    print("\n⚠️ Note: The BatchNorm error occurs because:")
    print("   - BatchNorm layers need multiple samples to compute statistics")
    print("   - Single graph testing fails due to insufficient batch size")
    print("   - In real training, graphs are batched together (batch_size=8)")
    print("   - This is why training works but single inference may fail")


def explain_graph_features():
    """Explain the graph features and structure"""
    print("\n🔄 GRAPH FEATURES EXPLANATION")
    print("=" * 50)

    code = """
def calculate_sum(a, b):
    result = a + b
    return result
"""
    print(f"Sample code: {code.strip()}")

    root = parse_code(code)
    if root:
        build_node_type_dict(root)
        graph = ast_to_graph(code)

        if graph:
            print(f"\n📊 Graph Structure:")
            print(f"   - Nodes: {graph.x.shape[0]}")
            print(f"   - Features per node: {graph.x.shape[1]}")
            print(f"   - Edges: {graph.edge_index.shape[1]}")

            print(f"\n🔍 Node Features (first 5 nodes):")
            for i in range(min(5, graph.x.shape[0])):
                features = graph.x[i]
                print(
                    f"   Node {i}: Type={features[0]:.0f}, Depth={features[1]:.0f}, Children={features[2]:.0f}")

            print(f"\n🔗 Edge Structure:")
            print(f"   Edge index shape: {graph.edge_index.shape}")
            print(f"   Each edge connects parent → child nodes")

            print(f"\n📈 Feature Explanation:")
            print(f"   - Type: Node type index (function_def, identifier, etc.)")
            print(f"   - Depth: How deep the node is in the AST")
            print(f"   - Children: Number of child nodes")


def main():
    """Main explanation function"""
    print("🧠 MODEL TRAINING EXPLANATION")
    print("=" * 60)
    print("This script explains how the model gets trained using data from the data/ folder")
    print("and how Python files are converted to graphs for training.\n")

    # Explain each part of the process
    explain_data_structure()
    demonstrate_file_loading()
    demonstrate_graph_conversion()
    demonstrate_training_data_preparation()
    demonstrate_model_training()
    explain_graph_features()

    print("\n" + "=" * 60)
    print("🎯 SUMMARY")
    print("The training process works as follows:")
    print("1. 📁 Load Python files from data/ folder")
    print("2. 🔍 Parse each file using Tree-sitter")
    print("3. 🌳 Convert valid code to AST, invalid code gets None")
    print("4. 🕸️ Convert AST to graph with node features")
    print("5. 🏷️ Label graphs: 0=valid, 1=invalid")
    print("6. 🧠 Train GNN model on graph batches")
    print("7. 💾 Save trained model as syntax_error_model.pth")

    print("\n✅ The model learns to distinguish between valid and invalid Python syntax")
    print("   by analyzing the graph structure of the code's AST!")


if __name__ == "__main__":
    main()
