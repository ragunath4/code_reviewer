#!/usr/bin/env python3
"""
Quick test script for syntax error classifier
"""

from syntax_error_classifier import SyntaxErrorClassifier

def test_examples():
    """Test with example code snippets"""
    classifier = SyntaxErrorClassifier()
    
    test_cases = [
        'def test():\n    pass',           # valid
        'def test()\n    pass',            # missing_colon
        'print("hello',                    # unclosed_string
        '  print("indented")',             # unexpected_indent
        'x = [1, 2, 3',                   # unexpected_eof
        'x = @invalid',                    # invalid_token
        'if x > 5:\n    print(x)',        # valid
        'for i in range(10):\n    print(i)'  # valid
    ]
    
    print("Testing Syntax Error Classifier")
    print("=" * 50)
    
    for i, code in enumerate(test_cases, 1):
        try:
            result = classifier.analyze_code(code)
            print(f"Test {i}: {result['result']} (conf: {result['confidence']:.2f})")
            print(f"Code: {repr(code)}")
            print("-" * 30)
        except Exception as e:
            print(f"Test {i}: Error - {e}")
            print("-" * 30)

if __name__ == "__main__":
    test_examples() 