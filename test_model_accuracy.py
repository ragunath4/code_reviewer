#!/usr/bin/env python3
"""
Model Accuracy Testing Script
Tests the syntax error classifier on various real-world broken code examples
"""

import torch
import torch.nn.functional as F
from syntax_error_classifier import SyntaxErrorClassifier, TokenBasedGraphBuilder
import json
import logging
from typing import List, Dict, Tuple
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelAccuracyTester:
    """Test the model's accuracy on various syntax error types"""
    
    def __init__(self):
        self.classifier = SyntaxErrorClassifier()
        self.graph_builder = TokenBasedGraphBuilder()
        
        # Define test cases with known expected results
        self.test_cases = {
            'valid': [
                'def test():\n    pass',
                'x = 5',
                'print("hello")',
                'if x > 5:\n    print(x)',
                'for i in range(10):\n    print(i)',
                'class MyClass:\n    def __init__(self):\n        pass',
                'try:\n    x = 1/0\nexcept:\n    pass',
                'with open("file.txt") as f:\n    pass',
                'def func(a, b):\n    return a + b',
                'x = [1, 2, 3]',
                'y = {"key": "value"}',
                'import os',
                'from pathlib import Path',
                'lambda x: x * 2',
                'x = [i for i in range(5)]',
                'assert x > 0',
                'raise ValueError("error")'
            ],
            'missing_colon': [
                'def test()\n    pass',
                'if x > 5\n    print(x)',
                'for i in range(10)\n    print(i)',
                'while True\n    break',
                'class MyClass\n    pass',
                'try\n    pass\nexcept\n    pass',
                'with open("file.txt")\n    pass',
                'elif x > 5\n    print(x)',
                'else\n    print("else")',
                'finally\n    pass'
            ],
            'unclosed_string': [
                'print("hello',
                'x = "unclosed string',
                'text = \'missing quote',
                'message = "Hello world',
                'path = "C:\\Users\\name',
                'data = "{"key": "value"',
                'sql = "SELECT * FROM table',
                'html = "<div class="container">',
                'json = \'{"name": "John"',
                'url = "https://example.com'
            ],
            'unexpected_indent': [
                '  print("indented")',
                '    x = 5',
                '  def func():\n    pass',
                '    if x > 5:\n      print(x)',
                '  class Test:\n    pass',
                '    try:\n      pass',
                '  for i in range(5):\n    print(i)',
                '    with open("file"):\n      pass',
                '  import os',
                '    from pathlib import Path'
            ],
            'unexpected_eof': [
                'x = [1, 2, 3',
                'y = {"key": "value"',
                'def func(a, b',
                'if x > 5 and y < 10',
                'for i in range(10',
                'while True and x > 5',
                'class MyClass(',
                'try:',
                'with open("file.txt"',
                'print("hello"',
                'x = (1 + 2',
                'y = [i for i in range(5',
                'z = {"a": 1, "b": 2',
                'def test(a, b, c',
                'if x > 5 or y < 10'
            ],
            'invalid_token': [
                'x = @invalid',
                'y = #comment',
                'z = $symbol',
                'a = &operator',
                'b = ^xor',
                'c = ~not',
                'd = `backtick',
                'e = |pipe',
                'f = \\backslash',
                'g = /slash',
                'h = *asterisk',
                'i = %modulo',
                'j = =equals',
                'k = +plus',
                'l = -minus'
            ]
        }
    
    def test_model_accuracy(self) -> Dict[str, Dict]:
        """Test model accuracy on all error types"""
        results = {}
        total_correct = 0
        total_tests = 0
        
        logger.info("Starting model accuracy testing...")
        logger.info("=" * 60)
        
        for error_type, test_cases in self.test_cases.items():
            logger.info(f"\nTesting {error_type.upper()} cases:")
            logger.info("-" * 40)
            
            correct = 0
            total = len(test_cases)
            predictions = []
            
            for i, code in enumerate(test_cases, 1):
                try:
                    result = self.classifier.analyze_code(code)
                    predicted_type = result['result']
                    confidence = result['confidence']
                    
                    is_correct = predicted_type == error_type
                    if is_correct:
                        correct += 1
                        total_correct += 1
                    
                    total_tests += 1
                    
                    predictions.append({
                        'code': code,
                        'expected': error_type,
                        'predicted': predicted_type,
                        'confidence': confidence,
                        'correct': is_correct
                    })
                    
                    status = "✓" if is_correct else "✗"
                    logger.info(f"{status} Test {i}: Expected '{error_type}', Got '{predicted_type}' (conf: {confidence:.2f})")
                    
                except Exception as e:
                    logger.error(f"Error testing case {i}: {e}")
                    total_tests += 1
            
            accuracy = (correct / total) * 100 if total > 0 else 0
            results[error_type] = {
                'accuracy': accuracy,
                'correct': correct,
                'total': total,
                'predictions': predictions
            }
            
            logger.info(f"Accuracy for {error_type}: {accuracy:.1f}% ({correct}/{total})")
        
        overall_accuracy = (total_correct / total_tests) * 100 if total_tests > 0 else 0
        results['overall'] = {
            'accuracy': overall_accuracy,
            'correct': total_correct,
            'total': total_tests
        }
        
        logger.info("\n" + "=" * 60)
        logger.info(f"OVERALL ACCURACY: {overall_accuracy:.1f}% ({total_correct}/{total_tests})")
        logger.info("=" * 60)
        
        return results
    
    def detailed_analysis(self, results: Dict[str, Dict]):
        """Provide detailed analysis of results"""
        logger.info("\nDETAILED ANALYSIS:")
        logger.info("=" * 60)
        
        # Confusion matrix analysis
        confusion_matrix = {}
        for error_type in self.test_cases.keys():
            confusion_matrix[error_type] = {}
            for predicted_type in self.test_cases.keys():
                confusion_matrix[error_type][predicted_type] = 0
        
        for error_type, data in results.items():
            if error_type == 'overall':
                continue
            for pred in data['predictions']:
                expected = pred['expected']
                predicted = pred['predicted']
                confusion_matrix[expected][predicted] += 1
        
        logger.info("\nConfusion Matrix:")
        logger.info("Expected → Predicted")
        logger.info("-" * 50)
        
        # Print header
        header = "Expected\\Predicted".ljust(20)
        for predicted_type in self.test_cases.keys():
            header += predicted_type[:8].ljust(10)
        logger.info(header)
        
        # Print matrix
        for expected_type in self.test_cases.keys():
            row = expected_type.ljust(20)
            for predicted_type in self.test_cases.keys():
                count = confusion_matrix[expected_type][predicted_type]
                row += str(count).ljust(10)
            logger.info(row)
        
        # Error analysis
        logger.info("\nERROR ANALYSIS:")
        logger.info("-" * 30)
        
        for error_type, data in results.items():
            if error_type == 'overall':
                continue
            
            incorrect_predictions = [p for p in data['predictions'] if not p['correct']]
            if incorrect_predictions:
                logger.info(f"\n{error_type.upper()} - Common misclassifications:")
                for pred in incorrect_predictions[:3]:  # Show top 3 errors
                    logger.info(f"  Expected: {pred['expected']}, Got: {pred['predicted']}")
                    logger.info(f"  Code: {repr(pred['code'])}")
                    logger.info(f"  Confidence: {pred['confidence']:.2f}")
    
    def save_results(self, results: Dict[str, Dict], filename: str = "model_accuracy_results.json"):
        """Save test results to JSON file"""
        # Convert torch tensors to lists for JSON serialization
        serializable_results = {}
        for key, value in results.items():
            if key == 'overall':
                serializable_results[key] = value
            else:
                serializable_results[key] = {
                    'accuracy': value['accuracy'],
                    'correct': value['correct'],
                    'total': value['total'],
                    'predictions': [
                        {
                            'code': p['code'],
                            'expected': p['expected'],
                            'predicted': p['predicted'],
                            'confidence': float(p['confidence']),
                            'correct': p['correct']
                        }
                        for p in value['predictions']
                    ]
                }
        
        with open(filename, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        logger.info(f"\nResults saved to {filename}")
    
    def test_real_world_examples(self):
        """Test on some real-world broken code examples"""
        real_world_examples = [
            {
                'code': 'def calculate_area(radius\n    return 3.14 * radius * radius',
                'expected': 'unexpected_eof',
                'description': 'Missing closing parenthesis in function definition'
            },
            {
                'code': 'if user_input == "yes"\n    print("Confirmed")\nelif user_input == "no"\n    print("Cancelled")',
                'expected': 'missing_colon',
                'description': 'Missing colons in if-elif statements'
            },
            {
                'code': 'data = {"name": "John", "age": 30, "city": "New York"',
                'expected': 'unexpected_eof',
                'description': 'Unclosed dictionary'
            },
            {
                'code': '    def helper_function():\n        return True',
                'expected': 'unexpected_indent',
                'description': 'Function defined with unexpected indentation'
            },
            {
                'code': 'message = "Hello, world!',
                'expected': 'unclosed_string',
                'description': 'Unclosed string literal'
            },
            {
                'code': 'x = @decorator\ndef func():\n    pass',
                'expected': 'invalid_token',
                'description': 'Invalid decorator syntax'
            }
        ]
        
        logger.info("\nREAL-WORLD EXAMPLES TESTING:")
        logger.info("=" * 50)
        
        correct = 0
        total = len(real_world_examples)
        
        for i, example in enumerate(real_world_examples, 1):
            try:
                result = self.classifier.analyze_code(example['code'])
                predicted_type = result['result']
                confidence = result['confidence']
                
                is_correct = predicted_type == example['expected']
                if is_correct:
                    correct += 1
                
                status = "✓" if is_correct else "✗"
                logger.info(f"{status} Example {i}: {example['description']}")
                logger.info(f"    Expected: {example['expected']}, Got: {predicted_type} (conf: {confidence:.2f})")
                logger.info(f"    Code: {repr(example['code'])}")
                logger.info()
                
            except Exception as e:
                logger.error(f"Error testing real-world example {i}: {e}")
        
        accuracy = (correct / total) * 100 if total > 0 else 0
        logger.info(f"Real-world accuracy: {accuracy:.1f}% ({correct}/{total})")


def main():
    """Main testing function"""
    tester = ModelAccuracyTester()
    
    # Run comprehensive accuracy testing
    results = tester.test_model_accuracy()
    
    # Provide detailed analysis
    tester.detailed_analysis(results)
    
    # Test real-world examples
    tester.test_real_world_examples()
    
    # Save results
    tester.save_results(results)
    
    logger.info("\nTesting completed!")


if __name__ == "__main__":
    main() 