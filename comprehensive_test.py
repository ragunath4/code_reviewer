import os
import torch
import numpy as np
from torch_geometric.data import DataLoader, Data
from graph_builder import ast_to_graph, NODE_TYPE_TO_IDX
from model import SyntaxGCN
from parser_util import parse_code
from sklearn.model_selection import train_test_split
import random


def test_model_generalization():
    print("=== GENERALIZATION TEST ===")

    # Create diverse test cases
    test_cases = [
        # Basic syntax errors
        ("missing_colon", "def test()\n    pass", 1),
        ("missing_paren", "print('hello'", 1),
        ("unclosed_string", "print('hello)", 1),
        # This might be valid in some contexts
        ("indentation_error", "def test():\npass", 0),
        ("invalid_indent", "def test():\n    pass\n  print('wrong')", 1),

        # Valid cases
        ("simple_print", "print('hello')", 0),
        ("function_def", "def test():\n    pass", 0),
        ("if_statement", "if True:\n    print('yes')", 0),
        ("for_loop", "for i in range(5):\n    print(i)", 0),

        # Edge cases
        ("empty_file", "", 0),
        ("only_comment", "# This is a comment", 0),
        ("import_statement", "import os", 0),
        ("class_def", "class Test:\n    pass", 0),

        # Complex cases
        ("nested_function", """
def outer():
    def inner():
        return 42
    return inner()
        """, 0),

        ("try_except", """
try:
    x = 1 / 0
except ZeroDivisionError:
    print('error')
        """, 0),

        ("list_comprehension", "[x for x in range(10)]", 0),

        # More syntax errors
        ("unclosed_bracket", "data = [1, 2, 3", 1),
        ("invalid_syntax", "def test():\n    return", 0),  # Valid
        ("missing_operator", "x = 5 6", 1),
        ("invalid_name", "1name = 5", 1),
    ]

    # Load and train model
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

    X_train, X_test, y_train, y_test, train_files, test_files = train_test_split(
        graphs, labels, filenames, test_size=0.3, random_state=42, stratify=labels)

    train_loader = DataLoader(X_train, batch_size=4, shuffle=True)

    model = SyntaxGCN(num_node_types=len(NODE_TYPE_TO_IDX))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = torch.nn.CrossEntropyLoss()

    # Train model
    for epoch in range(30):
        model.train()
        for batch in train_loader:
            optimizer.zero_grad()
            out = model(batch)
            loss = criterion(out, batch.y)
            loss.backward()
            optimizer.step()

    # Test on new cases
    model.eval()
    correct = 0
    total = 0

    print(f"{'Test Case':<25} | {'Expected':<8} | {'Predicted':<9} | {'Correct':<7} | {'Confidence':<10}")
    print("-" * 70)

    for name, code, expected in test_cases:
        try:
            # Parse code
            root = parse_code(code)
            if root is None:
                actual_expected = 1  # Syntax error
            else:
                actual_expected = 0  # Valid

            # Create graph
            if code.strip() == "":
                x = torch.tensor([[0, 0, 0]], dtype=torch.float)
                edge_index = torch.zeros((2, 0), dtype=torch.long)
                g = Data(x=x, edge_index=edge_index)
            else:
                g = ast_to_graph(code)

            g.batch = torch.zeros(g.x.size(0), dtype=torch.long)

            # Predict
            with torch.no_grad():
                out = model(g)
                pred = out.argmax(dim=1).item()
                prob = torch.softmax(out, dim=1).max().item()

            is_correct = pred == actual_expected
            if is_correct:
                correct += 1
            total += 1

            print(
                f"{name:<25} | {actual_expected:<8} | {pred:<9} | {'✓' if is_correct else '✗':<7} | {prob:.4f}")

        except Exception as e:
            print(
                f"{name:<25} | {expected:<8} | ERROR | {'✗':<7} | {str(e)[:20]}")
            total += 1

    print(
        f"\nGeneralization Accuracy: {correct/total*100:.2f}% ({correct}/{total})")


def test_data_quality():
    print("\n=== DATA QUALITY ANALYSIS ===")

    samples, labels, filenames = load_dataset()

    print("Dataset Statistics:")
    print(f"Total samples: {len(samples)}")
    print(f"Valid samples: {labels.count(0)}")
    print(f"Invalid samples: {labels.count(1)}")

    # Check for duplicates
    unique_samples = set()
    for sample in samples:
        if sample is not None:
            unique_samples.add(sample.strip())

    print(f"Unique samples: {len(unique_samples)}")
    print(f"Duplicate samples: {len(samples) - len(unique_samples)}")

    # Analyze graph characteristics
    graphs = build_graphs(samples)
    graph_sizes = [g.x.size(0) for g in graphs]

    print(f"\nGraph Analysis:")
    print(f"Average graph size: {np.mean(graph_sizes):.1f}")
    print(f"Min graph size: {min(graph_sizes)}")
    print(f"Max graph size: {max(graph_sizes)}")
    print(f"Graph size distribution: {sorted(graph_sizes)}")

    # Check for potential memorization
    print(f"\nPotential Issues:")
    print(f"1. Small dataset: {len(samples)} samples")
    print(
        f"2. Simple patterns: Most graphs are small ({np.mean(graph_sizes):.1f} nodes avg)")
    print(f"3. Binary classification: Only 2 classes")
    print(f"4. High confidence: Model might be overfitting")


def test_model_robustness():
    print("\n=== MODEL ROBUSTNESS TEST ===")

    # Test with noise/perturbations
    test_cases = [
        ("original", "def test():\n    return 42", 0),
        ("extra_space", "def test() :\n    return 42", 0),
        ("extra_newline", "def test():\n\n    return 42", 0),
        ("extra_comment", "def test():  # comment\n    return 42", 0),
        ("different_var", "def test():\n    return x", 0),
        ("different_number", "def test():\n    return 100", 0),
    ]

    # Load and train model
    samples, labels, filenames = load_dataset()

    for code in samples:
        if code is not None:
            root = parse_code(code)
            from graph_builder import build_node_type_dict
            build_node_type_dict(root)

    graphs = build_graphs(samples)
    for i, g in enumerate(graphs):
        g.y = torch.tensor([labels[i]], dtype=torch.long)
        g.batch = torch.zeros(g.x.size(0), dtype=torch.long)

    X_train, X_test, y_train, y_test, train_files, test_files = train_test_split(
        graphs, labels, filenames, test_size=0.3, random_state=42, stratify=labels)

    train_loader = DataLoader(X_train, batch_size=4, shuffle=True)

    model = SyntaxGCN(num_node_types=len(NODE_TYPE_TO_IDX))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = torch.nn.CrossEntropyLoss()

    # Train model
    for epoch in range(30):
        model.train()
        for batch in train_loader:
            optimizer.zero_grad()
            out = model(batch)
            loss = criterion(out, batch.y)
            loss.backward()
            optimizer.step()

    # Test robustness
    model.eval()
    predictions = []

    for name, code, expected in test_cases:
        try:
            g = ast_to_graph(code)
            g.batch = torch.zeros(g.x.size(0), dtype=torch.long)

            with torch.no_grad():
                out = model(g)
                pred = out.argmax(dim=1).item()
                prob = torch.softmax(out, dim=1).max().item()

            predictions.append((name, pred, prob))
            print(f"{name:<15} | Predicted: {pred} | Confidence: {prob:.4f}")

        except Exception as e:
            print(f"{name:<15} | Error: {str(e)}")

    # Check consistency
    if len(predictions) > 1:
        first_pred = predictions[0][1]
        consistent = all(pred[1] == first_pred for pred in predictions)
        print(f"\nPrediction consistency: {'✓' if consistent else '✗'}")


def load_dataset():
    samples = []
    labels = []
    filenames = []
    for fname in os.listdir('data'):
        if not fname.endswith('.py'):
            continue
        with open(os.path.join('data', fname), 'r', encoding='utf-8') as f:
            code = f.read()
        root = parse_code(code)
        if root is None:
            samples.append(None)
            labels.append(1)
        else:
            samples.append(code)
            labels.append(0)
        filenames.append(fname)
    return samples, labels, filenames


def build_graphs(samples):
    graphs = []
    for code in samples:
        if code is None:
            x = torch.tensor([[0, 0, 0]], dtype=torch.float)
            edge_index = torch.zeros((2, 0), dtype=torch.long)
            graphs.append(Data(x=x, edge_index=edge_index))
        else:
            g = ast_to_graph(code)
            graphs.append(g)
    return graphs


if __name__ == '__main__':
    test_model_generalization()
    test_data_quality()
    test_model_robustness()
