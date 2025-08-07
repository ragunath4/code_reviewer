#!/usr/bin/env python3
"""
Simple test script for the syntax error classifier
Can handle piped input or command line arguments
"""

import sys
from syntax_error_classifier import SyntaxErrorClassifier

def test_code(code):
    """Test a single piece of code"""
    try:
        classifier = SyntaxErrorClassifier()
        result = classifier.analyze_code(code)
        
        print(f"Code: {repr(code)}")
        print(f"Predicted: {result['result']}")
        print(f"Confidence: {result['confidence']:.2f}")
        print("-" * 50)
        
        return result
        
    except Exception as e:
        print(f"Error analyzing code: {e}")
        return None

def main():
    """Main function"""
    classifier = SyntaxErrorClassifier()
    
    # Check if input is piped
    if not sys.stdin.isatty():
        # Read from stdin (piped input)
        code = sys.stdin.read().strip()
        if code:
            test_code(code)
    else:
        # Interactive mode or command line argument
        if len(sys.argv) > 1:
            # Command line argument
            code = sys.argv[1]
            test_code(code)
        else:
            # Interactive mode
            print("Python Syntax Error Classifier")
            print("Enter code to test (or 'quit' to exit):")
            
            while True:
                try:
                    code = input("Code: ").strip()
                    if code.lower() == 'quit':
                        break
                    if code:
                        test_code(code)
                except KeyboardInterrupt:
                    print("\nExiting...")
                    break
                except EOFError:
                    break

if __name__ == "__main__":
    main() 