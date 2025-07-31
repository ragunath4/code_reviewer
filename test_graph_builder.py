from graph_builder import ast_to_graph
from parser_util import parse_code


def test_graph_builder():
    print("🧪 Testing Graph Builder with Updated Parser")
    print("=" * 50)

    # Test 1: Valid code - should create a graph
    valid_code = """
def greet(name):
    print("Hello", name)
"""
    print("\n1️⃣ Testing Valid Code Graph Creation:")
    print(f"Code: {valid_code.strip()}")

    # First test parser
    root = parse_code(valid_code)
    if root:
        print("✅ Parser: Valid code detected")

        # Then test graph builder
        graph = ast_to_graph(valid_code)
        if graph:
            print(f"✅ Graph Builder: Graph created successfully")
            print(f"   - Nodes: {graph.x.shape[0]}")
            print(f"   - Features per node: {graph.x.shape[1]}")
            print(f"   - Edges: {graph.edge_index.shape[1]}")
        else:
            print("❌ Graph Builder: Failed to create graph")
    else:
        print("❌ Parser: Incorrectly flagged valid code as invalid")

    # Test 2: Invalid code - should return None
    invalid_code = """
def greet(name)
    print("Hello", name)
"""
    print("\n2️⃣ Testing Invalid Code Graph Creation:")
    print(f"Code: {invalid_code.strip()}")

    # First test parser
    root = parse_code(invalid_code)
    if root:
        print("❌ Parser: Should have detected syntax error")
    else:
        print("✅ Parser: Correctly detected syntax error")

        # Then test graph builder
        graph = ast_to_graph(invalid_code)
        if graph is None:
            print("✅ Graph Builder: Correctly returned None for invalid code")
        else:
            print("❌ Graph Builder: Should have returned None for invalid code")

    # Test 3: Complex valid code
    complex_valid = """
   def add(x, y):   
        return x + y
    add(1, 2)
"""
    print("\n3️⃣ Testing Complex Valid Code Graph Creation:")
    print("Code: add function")

    root = parse_code(complex_valid)
    if root:
        print("✅ Parser: Complex valid code detected")

        graph = ast_to_graph(complex_valid)
        if graph:
            print(f"✅ Graph Builder: Complex graph created successfully")
            print(f"   - Nodes: {graph.x.shape[0]}")
            print(f"   - Features per node: {graph.x.shape[1]}")
            print(f"   - Edges: {graph.edge_index.shape[1]}")
        else:
            print("❌ Graph Builder: Failed to create complex graph")
    else:
        print("❌ Parser: Incorrectly flagged complex valid code as invalid")

    # Test 4: Nested error - should return None
    nested_error = """
def greet(name):
    if name:
        print("Hello", name
    return None
"""
    print("\n4️⃣ Testing Nested Error Graph Creation:")
    print(f"Code: {nested_error.strip()}")

    root = parse_code(nested_error)
    if root:
        print("❌ Parser: Should have detected nested syntax error")
    else:
        print("✅ Parser: Correctly detected nested syntax error")

        graph = ast_to_graph(nested_error)
        if graph is None:
            print("✅ Graph Builder: Correctly returned None for nested error")
        else:
            print("❌ Graph Builder: Should have returned None for nested error")

    print("\n" + "=" * 50)
    print("🎯 Graph Builder Test Summary:")
    print("The graph builder should:")
    print("- Create graphs for valid code")
    print("- Return None for invalid code")
    print("- Handle complex code structures")
    print("- Work seamlessly with the updated parser")


if __name__ == "__main__":
    test_graph_builder()
