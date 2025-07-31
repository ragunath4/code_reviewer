from parser_util import parse_code


def print_tree(node, indent=0):
    print("  " * indent +
          f"{node.type} [{node.start_point} - {node.end_point}]")
    for child in node.children:
        print_tree(child, indent + 1)


def test_parser():
    print("🧪 Testing Parser with Recursive Error Detection")
    print("=" * 50)

    # Test 1: Valid code
    valid_code = """
def greet(name):
    print("Hello", name)
"""
    print("\n1️⃣ Testing Valid Code:")
    print(f"Code: {valid_code.strip()}")
    result_valid = parse_code(valid_code)
    if result_valid:
        print("✅ Parser correctly: No syntax error detected")
    else:
        print("❌ Parser incorrectly flagged valid code as invalid")

    # Test 2: Invalid code - missing colon
    invalid_code1 = """
def greet(name)
    print("Hello", name)
"""
    print("\n2️⃣ Testing Invalid Code (Missing Colon):")
    print(f"Code: {invalid_code1.strip()}")
    result_invalid1 = parse_code(invalid_code1)
    if result_invalid1:
        print("❌ Parser missed syntax error (missing colon)")
    else:
        print("✅ Parser correctly detected syntax error")

    # Test 3: Invalid code - unclosed string
    invalid_code2 = """
def greet(name):
    print("Hello, name)
"""
    print("\n3️⃣ Testing Invalid Code (Unclosed String):")
    print(f"Code: {invalid_code2.strip()}")
    result_invalid2 = parse_code(invalid_code2)
    if result_invalid2:
        print("❌ Parser missed syntax error (unclosed string)")
    else:
        print("✅ Parser correctly detected syntax error")

    # Test 4: Invalid code - nested error in function
    invalid_code3 = """
def greet(name):
    if name:
        print("Hello", name
    return None
"""
    print("\n4️⃣ Testing Invalid Code (Nested Error - Missing Parenthesis):")
    print(f"Code: {invalid_code3.strip()}")
    result_invalid3 = parse_code(invalid_code3)
    if result_invalid3:
        print("❌ Parser missed nested syntax error")
    else:
        print("✅ Parser correctly detected nested syntax error")

    # Test 5: Invalid code - indentation error
    invalid_code4 = """
def greet(name):
print("Hello", name)
"""
    print("\n5️⃣ Testing Invalid Code (Indentation Error):")
    print(f"Code: {invalid_code4.strip()}")
    result_invalid4 = parse_code(invalid_code4)
    if result_invalid4:
        print("❌ Parser missed indentation error")
    else:
        print("✅ Parser correctly detected indentation error")

    # Test 6: Complex valid code
    complex_valid = """
class Calculator:
    def __init__(self):
        self.result = 0
    
    def add(self, x, y):
        return x + y
    
    def multiply(self, x, y):
        return x * y
"""
    print("\n6️⃣ Testing Complex Valid Code:")
    print("Code: Calculator class with methods")
    result_complex = parse_code(complex_valid)
    if result_complex:
        print("✅ Parser correctly: No syntax error in complex code")
    else:
        print("❌ Parser incorrectly flagged complex valid code as invalid")

    print("\n" + "=" * 50)
    print("🎯 Parser Test Summary:")
    print("The parser should detect errors in child nodes, not just root nodes.")
    print("This ensures accurate syntax error detection for the GCN model.")


if __name__ == "__main__":
    test_parser()
