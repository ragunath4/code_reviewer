#!/usr/bin/env python3
"""
Interactive Testing Script
Allows users to input their own code and see the model's predictions
"""

import torch
import sys
from syntax_error_classifier import SyntaxErrorClassifier
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_single_code():
    """Test a single piece of code"""
    classifier = SyntaxErrorClassifier()

    print("\n" + "="*60)
    print("INTERACTIVE SYNTAX ERROR CLASSIFIER")
    print("="*60)
    print("Enter your Python code (press Enter twice to finish):")
    print("Type 'quit' to exit")
    print("-"*60)

    while True:
        try:
            # Get code input
            print("\nEnter code:")
            lines = []
            while True:
                try:
                    line = input()
                    if line.strip() == 'quit':
                        return
                    if line.strip() == '':
                        break
                    lines.append(line)
                except EOFError:
                    # Handle piped input
                    if lines:
                        break
                    else:
                        return

            if not lines:
                continue

            code = '\n'.join(lines)

            # Analyze the code
            print("\n" + "-"*40)
            print("ANALYSIS RESULTS:")
            print("-"*40)

            result = classifier.analyze_code(code)

            print(f"Predicted Error Type: {result['result']}")
            print(f"Confidence: {result['confidence']:.2f}")

            # Provide explanation
            error_type = result['result']
            confidence = result['confidence']

            print(f"\nExplanation:")
            if error_type == 'valid':
                print("✓ Code appears to be syntactically valid")
            elif error_type == 'missing_colon':
                print("✗ Missing colon (:) after control flow statements")
                print(
                    "   Common in: if, elif, else, for, while, def, class, try, except, finally, with")
            elif error_type == 'unclosed_string':
                print("✗ Unclosed string literal")
                print("   Check for missing quotes at the end of strings")
            elif error_type == 'unexpected_indent':
                print("✗ Unexpected indentation")
                print("   Code is indented without proper context (missing colon, etc.)")
            elif error_type == 'unexpected_eof':
                print("✗ Unexpected end of file")
                print("   Missing closing brackets, parentheses, or braces")
            elif error_type == 'invalid_token':
                print("✗ Invalid token or character")
                print("   Contains characters not valid in Python syntax")

            if confidence < 0.5:
                print(
                    f"\n⚠️  Low confidence ({confidence:.2f}) - prediction may be unreliable")
            elif confidence > 0.8:
                print(
                    f"\n✅ High confidence ({confidence:.2f}) - prediction is likely accurate")
            else:
                print(
                    f"\n⚠️  Medium confidence ({confidence:.2f}) - prediction should be verified")

            print("\n" + "="*60)

        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"\nError analyzing code: {e}")
            print("Please try again.")


def batch_test():
    """Test multiple code snippets from a file"""
    classifier = SyntaxErrorClassifier()

    print("\n" + "="*60)
    print("BATCH TESTING MODE")
    print("="*60)
    print("Enter file path containing code snippets (one per line):")
    print("Or type 'demo' for example tests")

    try:
        file_path = input("File path: ").strip()
    except EOFError:
        print("No input provided. Exiting...")
        return

    if file_path.lower() == 'demo':
        # Run demo tests
        demo_tests = [
            'def test():\n    pass',
            'print("hello',
            'if x > 5\n    print(x)',
            'x = [1, 2, 3',
            '  print("indented")',
            'x = @invalid',
            'def valid():\n    return True'
        ]

        print("\nRunning demo tests...")
        print("-"*40)

        for i, code in enumerate(demo_tests, 1):
            try:
                result = classifier.analyze_code(code)
                print(
                    f"Test {i}: {result['result']} (conf: {result['confidence']:.2f})")
                print(f"Code: {repr(code)}")
                print()
            except Exception as e:
                print(f"Test {i}: Error - {e}")
                print()

        return

    try:
        with open(file_path, 'r') as f:
            lines = f.readlines()

        print(f"\nTesting {len(lines)} code snippets...")
        print("-"*40)

        results = []
        for i, line in enumerate(lines, 1):
            code = line.strip()
            if not code or code.startswith('#'):
                continue

            try:
                result = classifier.analyze_code(code)
                results.append({
                    'line': i,
                    'code': code,
                    'predicted': result['result'],
                    'confidence': result['confidence']
                })

                status = "✓" if result['result'] == 'valid' else "✗"
                print(
                    f"{status} Line {i}: {result['result']} (conf: {result['confidence']:.2f})")

            except Exception as e:
                print(f"✗ Line {i}: Error - {e}")

        # Summary
        print("\n" + "="*40)
        print("SUMMARY:")
        print("="*40)

        error_counts = {}
        for result in results:
            error_type = result['predicted']
            error_counts[error_type] = error_counts.get(error_type, 0) + 1

        for error_type, count in error_counts.items():
            print(f"{error_type}: {count}")

        valid_count = error_counts.get('valid', 0)
        total_count = len(results)
        print(
            f"\nValid code: {valid_count}/{total_count} ({valid_count/total_count*100:.1f}%)")

    except FileNotFoundError:
        print(f"File '{file_path}' not found!")
    except Exception as e:
        print(f"Error reading file: {e}")


def main():
    """Main function"""
    print("Python Syntax Error Classifier - Interactive Testing")
    print("="*60)
    print("Choose testing mode:")
    print("1. Interactive (input code manually)")
    print("2. Batch testing (from file)")
    print("3. Exit")

    while True:
        try:
            choice = input("\nEnter choice (1-3): ").strip()

            if choice == '1':
                test_single_code()
            elif choice == '2':
                batch_test()
            elif choice == '3':
                print("Goodbye!")
                break
            else:
                print("Invalid choice. Please enter 1, 2, or 3.")

        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except EOFError:
            print("\nNo input provided. Exiting...")
            break
        except Exception as e:
            print(f"Error: {e}")


if __name__ == "__main__":
    main()
