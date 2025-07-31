import torch
from graph_builder import ast_to_graph
from parser_util import parse_code
from enhanced_model import EnhancedSyntaxGCN


def test_gnn():
    print("🧪 Testing GNN Model with Updated Parser and Graph Builder")
    print("=" * 60)

    # Initialize the model
    model = EnhancedSyntaxGCN(
        num_node_types=50,  # Approximate number of node types
        hidden_dim=64,
        num_layers=3
    )

    print(f"✅ Model initialized: {type(model).__name__}")

    # Test 1: Valid code
    valid_code = """
def greet(name):
    print("Hello", name)
"""
    print("\n1️⃣ Testing Valid Code with GNN:")
    print(f"Code: {valid_code.strip()}")

    graph = ast_to_graph(valid_code)
    if graph:
        print(
            f"✅ Graph created: {graph.x.shape[0]} nodes, {graph.edge_index.shape[1]} edges")

        # Add batch dimension
        graph.batch = torch.zeros(graph.x.shape[0], dtype=torch.long)

        # Forward pass
        with torch.no_grad():
            output = model(graph)
            probabilities = torch.softmax(output, dim=1)
            prediction = torch.argmax(probabilities, dim=1)
            confidence = torch.max(probabilities, dim=1)[0]

        print(
            f"✅ GNN Prediction: {'Valid' if prediction.item() == 0 else 'Invalid'}")
        print(f"   Confidence: {confidence.item():.3f}")
        print(f"   Raw output: {output.item():.3f}")
    else:
        print("❌ Failed to create graph for valid code")

    # Test 2: Invalid code
    invalid_code = """
def greet(name)
    print("Hello", name)
"""
    print("\n2️⃣ Testing Invalid Code with GNN:")
    print(f"Code: {invalid_code.strip()}")

    graph = ast_to_graph(invalid_code)
    if graph is None:
        print("✅ Graph builder correctly returned None for invalid code")
        print("   This prevents GNN from processing invalid syntax")
    else:
        print("❌ Graph builder should have returned None")

    # Test 3: Complex valid code
    complex_valid = """
class Calculator:
    def __init__(self):
        self.result = 0
    
    def add(self, x, y):
        return x + y
"""
    print("\n3️⃣ Testing Complex Valid Code with GNN:")
    print("Code: Calculator class with methods")

    graph = ast_to_graph(complex_valid)
    if graph:
        print(
            f"✅ Complex graph created: {graph.x.shape[0]} nodes, {graph.edge_index.shape[1]} edges")

        # Add batch dimension
        graph.batch = torch.zeros(graph.x.shape[0], dtype=torch.long)

        # Forward pass
        with torch.no_grad():
            output = model(graph)
            probabilities = torch.softmax(output, dim=1)
            prediction = torch.argmax(probabilities, dim=1)
            confidence = torch.max(probabilities, dim=1)[0]

        print(
            f"✅ GNN Prediction: {'Valid' if prediction.item() == 0 else 'Invalid'}")
        print(f"   Confidence: {confidence.item():.3f}")
        print(f"   Raw output: {output.item():.3f}")
    else:
        print("❌ Failed to create complex graph")

    # Test 4: Nested error
    nested_error = """
def greet(name):
    if name:
        print("Hello", name
    return None
"""
    print("\n4️⃣ Testing Nested Error with GNN:")
    print(f"Code: {nested_error.strip()}")

    graph = ast_to_graph(nested_error)
    if graph is None:
        print("✅ Graph builder correctly returned None for nested error")
        print("   This prevents GNN from processing invalid syntax")
    else:
        print("❌ Graph builder should have returned None")

    print("\n" + "=" * 60)
    print("🎯 GNN Test Summary:")
    print("The GNN model should:")
    print("- Process valid code graphs successfully")
    print("- Not process invalid code (graph builder returns None)")
    print("- Handle complex code structures")
    print("- Work seamlessly with parser and graph builder")
    print("- Provide confidence scores for predictions")


if __name__ == "__main__":
    test_gnn()
