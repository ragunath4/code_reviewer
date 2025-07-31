#!/usr/bin/env python3
"""
Demo Script for Syntax Error Analyzer
Demonstrates the complete workflow of training and using the syntax analyzer
"""

import os
import sys
import subprocess
import time
from syntax_analyzer import SyntaxAnalyzer

def print_header(title):
    """Print a formatted header"""
    print("\n" + "="*60)
    print(f" {title}")
    print("="*60)

def print_subheader(title):
    """Print a formatted subheader"""
    print(f"\n--- {title} ---")

def check_dependencies():
    """Check if required dependencies are installed"""
    print_header("DEPENDENCY CHECK")
    
    required_packages = [
        'torch',
        'torch_geometric', 
        'tree_sitter_python',
        'sklearn'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package} - NOT INSTALLED")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n❌ Missing packages: {', '.join(missing_packages)}")
        print("Please install them using: pip install -r requirements.txt")
        return False
    
    print("\n✅ All dependencies are installed!")
    return True

def check_dataset():
    """Check if the dataset exists"""
    print_header("DATASET CHECK")
    
    data_dir = 'data'
    if not os.path.exists(data_dir):
        print(f"❌ Dataset directory '{data_dir}' not found!")
        return False
    
    files = [f for f in os.listdir(data_dir) if f.endswith('.py')]
    valid_files = [f for f in files if f.startswith('valid_')]
    invalid_files = [f for f in files if f.startswith('invalid_')]
    
    print(f"📁 Dataset directory: {data_dir}")
    print(f"📄 Total Python files: {len(files)}")
    print(f"✅ Valid code samples: {len(valid_files)}")
    print(f"❌ Invalid code samples: {len(invalid_files)}")
    
    if len(files) < 10:
        print("⚠️  Warning: Small dataset may affect model performance")
    
    return True

def train_model():
    """Train the model"""
    print_header("MODEL TRAINING")
    
    if os.path.exists('syntax_error_model.pth'):
        print("✅ Trained model already exists: syntax_error_model.pth")
        return True
    
    print("🚀 Starting model training...")
    print("This may take a few minutes...")
    
    try:
        # Run the training script
        result = subprocess.run([sys.executable, 'train_model.py'], 
                              capture_output=True, text=True, timeout=300)
        
        if result.returncode == 0:
            print("✅ Model training completed successfully!")
            return True
        else:
            print(f"❌ Training failed: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        print("❌ Training timed out (5 minutes)")
        return False
    except Exception as e:
        print(f"❌ Training error: {e}")
        return False

def test_analyzer():
    """Test the syntax analyzer with various examples"""
    print_header("ANALYZER TESTING")
    
    # Initialize analyzer
    analyzer = SyntaxAnalyzer()
    
    # Test cases
    test_cases = [
        {
            'name': 'Valid Function',
            'code': '''def calculate_sum(a, b):
    result = a + b
    return result

total = calculate_sum(10, 20)
print(f"Sum: {total}")''',
            'expected': 'valid'
        },
        {
            'name': 'Missing Colon Error',
            'code': '''def calculate_sum(a, b)
    result = a + b
    return result

total = calculate_sum(10, 20)
print(f"Sum: {total}")''',
            'expected': 'invalid'
        },
        {
            'name': 'Indentation Error',
            'code': '''def calculate_sum(a, b):
    result = a + b
return result

total = calculate_sum(10, 20)
print(f"Sum: {total}")''',
            'expected': 'invalid'
        },
        {
            'name': 'Missing Parenthesis',
            'code': '''def calculate_sum(a, b:
    result = a + b
    return result

total = calculate_sum(10, 20
print(f"Sum: {total}")''',
            'expected': 'invalid'
        },
        {
            'name': 'Unclosed String',
            'code': '''def greet(name):
    message = "Hello, " + name
    print(message

greet("World")''',
            'expected': 'invalid'
        },
        {
            'name': 'Complex Valid Code',
            'code': '''class Calculator:
    def __init__(self):
        self.history = []
    
    def add(self, a, b):
        result = a + b
        self.history.append(f"{a} + {b} = {result}")
        return result
    
    def get_history(self):
        return self.history

calc = Calculator()
result = calc.add(5, 3)
print(f"Result: {result}")
print(f"History: {calc.get_history()}")''',
            'expected': 'valid'
        }
    ]
    
    results = []
    correct = 0
    
    for i, case in enumerate(test_cases, 1):
        print_subheader(f"Test {i}: {case['name']}")
        
        try:
            result = analyzer.analyze_code(case['code'])
            predicted = "invalid" if result['has_syntax_error'] else "valid"
            is_correct = predicted == case['expected']
            
            if is_correct:
                correct += 1
                print(f"✅ CORRECT: Expected {case['expected']}, Got {predicted}")
            else:
                print(f"❌ WRONG: Expected {case['expected']}, Got {predicted}")
            
            print(f"   Confidence: {result['confidence']:.1%}")
            print(f"   Error Type: {result['error_type']}")
            
            results.append({
                'case': case['name'],
                'expected': case['expected'],
                'predicted': predicted,
                'correct': is_correct,
                'confidence': result['confidence'],
                'error_type': result['error_type']
            })
            
        except Exception as e:
            print(f"❌ ERROR: {e}")
            results.append({
                'case': case['name'],
                'expected': case['expected'],
                'predicted': 'error',
                'correct': False,
                'confidence': 0,
                'error_type': 'analysis_error'
            })
    
    # Summary
    accuracy = correct / len(test_cases) * 100
    print_subheader("TEST RESULTS SUMMARY")
    print(f"Total Tests: {len(test_cases)}")
    print(f"Correct Predictions: {correct}")
    print(f"Accuracy: {accuracy:.1f}%")
    
    return results

def interactive_demo():
    """Interactive demo mode"""
    print_header("INTERACTIVE DEMO")
    print("Enter Python code to analyze (type 'quit' to exit)")
    print("Press Enter twice to analyze the code")
    
    analyzer = SyntaxAnalyzer()
    
    while True:
        print("\n" + "-"*40)
        print("Enter your Python code:")
        
        lines = []
        while True:
            try:
                line = input()
                if line.lower() == 'quit':
                    return
                if line == "" and lines and lines[-1] == "":
                    break
                lines.append(line)
            except KeyboardInterrupt:
                print("\nExiting...")
                return
        
        code = '\n'.join(lines[:-1])
        if code.strip():
            try:
                result = analyzer.analyze_code(code)
                analyzer.print_analysis(result, code)
                
                # Save result
                with open('interactive_result.json', 'w') as f:
                    import json
                    json.dump(result, f, indent=2)
                print("Result saved to: interactive_result.json")
                
            except Exception as e:
                print(f"❌ Analysis error: {e}")

def show_usage_examples():
    """Show usage examples"""
    print_header("USAGE EXAMPLES")
    
    examples = [
        {
            'description': 'Analyze a Python file',
            'command': 'python syntax_analyzer.py --file my_script.py'
        },
        {
            'description': 'Analyze code from command line',
            'command': 'python syntax_analyzer.py --code "def hello(): print(\'world\')"'
        },
        {
            'description': 'Save results to custom file',
            'command': 'python syntax_analyzer.py --file test.py --output results.json'
        },
        {
            'description': 'Use custom model file',
            'command': 'python syntax_analyzer.py --file test.py --model my_model.pth'
        },
        {
            'description': 'Verbose output',
            'command': 'python syntax_analyzer.py --file test.py --verbose'
        }
    ]
    
    for i, example in enumerate(examples, 1):
        print(f"{i}. {example['description']}")
        print(f"   {example['command']}")
        print()

def main():
    """Main demo function"""
    print_header("SYNTAX ERROR ANALYZER DEMO")
    print("This demo will guide you through the complete workflow")
    
    # Step 1: Check dependencies
    if not check_dependencies():
        print("\n❌ Please install missing dependencies and try again")
        return 1
    
    # Step 2: Check dataset
    if not check_dataset():
        print("\n❌ Dataset not found. Please ensure the 'data' directory exists with Python files")
        return 1
    
    # Step 3: Train model (if needed)
    if not train_model():
        print("\n❌ Model training failed")
        return 1
    
    # Step 4: Test analyzer
    test_results = test_analyzer()
    
    # Step 5: Show usage examples
    show_usage_examples()
    
    # Step 6: Interactive demo
    print_header("INTERACTIVE DEMO")
    print("Would you like to try the interactive demo? (y/n)")
    
    try:
        response = input().lower().strip()
        if response in ['y', 'yes']:
            interactive_demo()
    except KeyboardInterrupt:
        print("\nDemo interrupted")
    
    print_header("DEMO COMPLETED")
    print("✅ You can now use the syntax analyzer!")
    print("📖 Check the documentation for more details")
    
    return 0

if __name__ == '__main__':
    exit(main()) 