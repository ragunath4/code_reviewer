#!/usr/bin/env python3
"""
Demo script for the Python Syntax Error Classifier
Shows the model working with various examples
"""

from syntax_error_classifier import SyntaxErrorClassifier

def run_demo():
    """Run a demonstration of the syntax error classifier"""
    classifier = SyntaxErrorClassifier()
    
    print("Python Syntax Error Classifier - Demo")
    print("=" * 60)
    print("Testing various code examples...")
    print()
    
    examples = [
        {
            "code": "def test():\n    pass",
            "description": "Valid function definition"
        },
        {
            "code": "def test()\n    pass", 
            "description": "Missing colon after function definition"
        },
        {
            "code": "print(\"hello",
            "description": "Unclosed string literal"
        },
        {
            "code": "  print(\"indented\")",
            "description": "Unexpected indentation"
        },
        {
            "code": "x = [1, 2, 3",
            "description": "Unclosed list (unexpected EOF)"
        },
        {
            "code": "x = @invalid",
            "description": "Invalid token (@)"
        },
        {
            "code": "if x > 5:\n    print(x)",
            "description": "Valid if statement"
        },
        {
            "code": "for i in range(10):\n    print(i)",
            "description": "Valid for loop"
        },
        {
            "code": "class MyClass:\n    def __init__(self):\n        pass",
            "description": "Valid class definition"
        },
        {
            "code": "try:\n    x = 1/0\nexcept:\n    pass",
            "description": "Valid try-except block"
        }
    ]
    
    results = []
    
    for i, example in enumerate(examples, 1):
        code = example["code"]
        description = example["description"]
        
        try:
            result = classifier.analyze_code(code)
            
            # Determine if prediction is correct
            expected = "valid" if "Valid" in description else "error"
            is_correct = (expected == "valid" and result["result"] == "valid") or \
                        (expected == "error" and result["result"] != "valid")
            
            status = "✅" if is_correct else "❌"
            
            print(f"Test {i}: {status}")
            print(f"Code: {repr(code)}")
            print(f"Description: {description}")
            print(f"Predicted: {result['result']} (confidence: {result['confidence']:.2f})")
            print("-" * 50)
            
            results.append({
                "test": i,
                "code": code,
                "description": description,
                "predicted": result["result"],
                "confidence": result["confidence"],
                "correct": is_correct
            })
            
        except Exception as e:
            print(f"Test {i}: Error - {e}")
            print("-" * 50)
    
    # Summary
    print("\n" + "=" * 60)
    print("DEMO SUMMARY")
    print("=" * 60)
    
    correct = sum(1 for r in results if r["correct"])
    total = len(results)
    
    print(f"Total tests: {total}")
    print(f"Correct predictions: {correct}")
    print(f"Accuracy: {correct/total*100:.1f}%")
    
    # Error type breakdown
    error_types = {}
    for result in results:
        error_type = result["predicted"]
        error_types[error_type] = error_types.get(error_type, 0) + 1
    
    print(f"\nError type breakdown:")
    for error_type, count in error_types.items():
        print(f"  {error_type}: {count}")
    
    print("\nDemo completed!")

if __name__ == "__main__":
    run_demo() 