#!/usr/bin/env python3
"""
Script for testing syntax error classifier with piped input
"""

import sys
from syntax_error_classifier import SyntaxErrorClassifier


def main():
    """Main function"""
    classifier = SyntaxErrorClassifier()

    # Read input from stdin
    if not sys.stdin.isatty():
        # Piped input
        code = sys.stdin.read().strip()
        if code:
            # Replace literal \n with actual newlines
            code = code.replace('\\n', '\n')

            try:
                result = classifier.analyze_code(code)
                print(f"Code: {repr(code)}")
                print(f"Predicted: {result['result']}")
                print(f"Confidence: {result['confidence']:.2f}")
            except Exception as e:
                print(f"Error: {e}")
    else:
        print("No piped input detected. Use: echo 'code' | python pipe_test.py")


if __name__ == "__main__":
    main()
