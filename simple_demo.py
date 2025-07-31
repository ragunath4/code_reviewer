#!/usr/bin/env python3
"""
Simple Demo for Syntax Error Detector
Works without complex dependencies
"""

import json
import time
import os

def analyze_code_simple(code):
    lines = code.split('\n')
    errors = []
    confidence = 0.8

    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith(('def ', 'class ', 'if ', 'for ', 'while ', 'try:', 'except:', 'finally:', 'else:', 'elif ')):
            if not stripped.endswith(':'):
                errors.append(f"Missing colon on line {i+1}")
                confidence = 0.95

    for i, line in enumerate(lines):
        if line.strip() and not line.startswith(' ') and i > 0:
            prev_line = lines[i-1].strip()
            if prev_line.endswith(':'):
                errors.append(f"Indentation error on line {i+1}")
                confidence = 0.9

    open_paren = code.count('(')
    close_paren = code.count(')')
    open_bracket = code.count('[')
    close_bracket = code.count(']')
    open_brace = code.count('{')
    close_brace = code.count('}')

    if open_paren != close_paren:
        errors.append("Missing closing parenthesis")
        confidence = 0.9
    if open_bracket != close_bracket:
        errors.append("Missing closing bracket")
        confidence = 0.9
    if open_brace != close_brace:
        errors.append("Missing closing brace")
        confidence = 0.9

    single_quotes = code.count("'")
    double_quotes = code.count('"')
    if single_quotes % 2 != 0 or double_quotes % 2 != 0:
        errors.append("Unclosed string literal")
        confidence = 0.85

    for line in lines:
        if line.strip() and '=' in line:
            var_name = line.split('=')[0].strip()
            if var_name and var_name[0].isdigit():
                errors.append("Invalid variable name (starts with number)")
                confidence = 0.8

    has_error = len(errors) > 0
    error_type = errors[0] if errors else "valid_syntax"

    return {
        'has_syntax_error': has_error,
        'confidence': confidence,
        'error_type': error_type,
        'message': "Syntax error detected" if has_error else "Code is syntactically correct",
        'details': f"Found {len(errors)} error(s): {'; '.join(errors)}" if errors else f"Confidence: {confidence:.2%}",
        'code_length': len(code),
        'ast_nodes': len(code.split())
    }

def demo_examples():
    demo_cases = [
        {
            "name": "Valid Function",
            "code": """def calculate_sum(a, b):\n    result = a + b\n    return result\n\ntotal = calculate_sum(10, 20)\nprint(f\"Sum: {total}\")""",
            "expected": "valid"
        },
        {
            "name": "Missing Colon Error",
            "code": """def calculate_sum(a, b)\n    result = a + b\n    return result\n\ntotal = calculate_sum(10, 20)\nprint(f\"Sum: {total}\")""",
            "expected": "invalid"
        },
        {
            "name": "Indentation Error",
            "code": """def calculate_sum(a, b):\n    result = a + b\nreturn result\n\ntotal = calculate_sum(10, 20)\nprint(f\"Sum: {total}\")""",
            "expected": "invalid"
        },
        {
            "name": "Missing Parenthesis",
            "code": """def calculate_sum(a, b:\n    result = a + b\n    return result\n\ntotal = calculate_sum(10, 20\nprint(f\"Sum: {total}\")""",
            "expected": "invalid"
        },
        {
            "name": "Complex Valid Code",
            "code": """class Calculator:\n    def __init__(self):\n        self.history = []\n\n    def add(self, a, b):\n        result = a + b\n        self.history.append(f\"{a} + {b} = {result}\")\n        return result\n\n    def get_history(self):\n        return self.history\n\ncalc = Calculator()\nresult = calc.add(5, 3)\nprint(f\"Result: {result}\")\nprint(f\"History: {calc.get_history()}\")""",
            "expected": "valid"
        },
        {
            "name": "Complex Error",
            "code": """class Calculator:\n    def __init__(self):\n        self.history = []\n\n    def add(self, a, b\n        result = a + b\n        self.history.append(f\"{a} + {b} = {result}\")\n        return result\n\n    def get_history(self):\n        return self.history\n\ncalc = Calculator()\nresult = calc.add(5, 3\nprint(f\"Result: {result}\")""",
            "expected": "invalid"
        }
    ]

    results = []
    for case in demo_cases:
        result = analyze_code_simple(case['code'])
        predicted = "invalid" if result['has_syntax_error'] else "valid"
        results.append({
            'case': case['name'],
            'expected': case['expected'],
            'predicted': predicted,
            'correct': predicted == case['expected'],
            'confidence': result['confidence'],
            'error_type': result['error_type']
        })

    return results

def show_statistics(results):
    total = len(results)
    correct = sum(1 for r in results if r['correct'])
    accuracy = correct / total * 100
    confidences = [r['confidence'] for r in results]
    avg_confidence = sum(confidences) / len(confidences)
    error_types = {}
    for r in results:
        error_type = r['error_type']
        error_types[error_type] = error_types.get(error_type, 0) + 1

    stats = {
        'total': total,
        'correct': correct,
        'accuracy': accuracy,
        'avg_confidence': avg_confidence,
        'min_confidence': min(confidences),
        'max_confidence': max(confidences),
        'error_type_distribution': error_types
    }
    with open("demo_statistics.json", "w") as f:
        json.dump(stats, f, indent=2)

def interactive_demo():
    while True:
        lines = []
        while True:
            line = input()
            if line.lower() == 'quit':
                return
            if line == "" and lines and lines[-1] == "":
                break
            lines.append(line)
        code = '\n'.join(lines[:-1])
        if code.strip():
            result = analyze_code_simple(code)
            with open('interactive_result.json', 'w') as f:
                json.dump(result, f, indent=2)

def main():
    results = demo_examples()
    show_statistics(results)
    interactive_demo()

if __name__ == '__main__':
    main()
