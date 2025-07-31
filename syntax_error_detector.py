#!/usr/bin/env python3
"""
Syntax Error Detector - User Interface
Takes Python code as input and predicts whether it has syntax errors
"""

import os
import torch
import json
from torch_geometric.data import Data
from graph_builder import ast_to_graph, NODE_TYPE_TO_IDX
from enhanced_model import EnhancedSyntaxGCN
from parser_util import parse_code
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SyntaxErrorDetector:
    def __init__(self, model_path='syntax_error_model.pth'):
        """Initialize the syntax error detector"""
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        # Load model if exists, otherwise create new
        if os.path.exists(model_path):
            self.model = EnhancedSyntaxGCN(num_node_types=len(NODE_TYPE_TO_IDX)).to(self.device)
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            logger.info(f"Loaded model from {model_path}")
        else:
            logger.warning(f"Model not found at {model_path}. Please train the model first.")
            self.model = None
    
    def analyze_code(self, code: str) -> dict:
        """
        Analyze Python code for syntax errors
        
        Args:
            code (str): Python code to analyze
            
        Returns:
            dict: Analysis results with prediction and confidence
        """
        try:
            # Parse the code
            root = parse_code(code)
            
            # Check if parsing failed (syntax error)
            if root is None:
                return {
                    'has_syntax_error': True,
                    'confidence': 1.0,
                    'error_type': 'Parsing failed',
                    'message': 'Code contains syntax errors that prevent parsing',
                    'details': 'The code could not be parsed by Tree-sitter, indicating syntax errors'
                }
            
            # Convert to graph
            graph = ast_to_graph(code)
            graph.batch = torch.zeros(graph.x.size(0), dtype=torch.long)
            graph = graph.to(self.device)
            
            # Make prediction
            if self.model is None:
                return {
                    'has_syntax_error': False,
                    'confidence': 0.5,
                    'error_type': 'Model not available',
                    'message': 'Model not trained yet',
                    'details': 'Please train the model first using improved_trainer.py'
                }
            
            self.model.eval()
            with torch.no_grad():
                output = self.model(graph)
                probabilities = torch.softmax(output, dim=1)
                prediction = output.argmax(dim=1).item()
                confidence = probabilities.max().item()
            
            # Interpret results
            has_error = prediction == 1
            error_types = {
                'missing_colon': 'Missing colon after function/class definition',
                'indentation_error': 'Incorrect indentation',
                'missing_paren': 'Missing closing parenthesis',
                'unclosed_string': 'Unclosed string literal',
                'invalid_syntax': 'Invalid syntax structure'
            }
            
            # Determine error type based on code analysis
            error_type = self._detect_error_type(code)
            
            return {
                'has_syntax_error': has_error,
                'confidence': confidence,
                'error_type': error_type,
                'message': self._get_message(has_error, error_type),
                'details': self._get_details(has_error, error_type, confidence),
                'code_length': len(code),
                'ast_nodes': graph.x.size(0) if hasattr(graph, 'x') else 0
            }
            
        except Exception as e:
            logger.error(f"Error analyzing code: {e}")
            return {
                'has_syntax_error': True,
                'confidence': 0.8,
                'error_type': 'Analysis error',
                'message': f'Error during analysis: {str(e)}',
                'details': 'The code could not be properly analyzed due to an error in the analysis process'
            }
    
    def _detect_error_type(self, code: str) -> str:
        """Detect specific error type based on code patterns"""
        lines = code.split('\n')
        
        # Check for missing colons
        for line in lines:
            if line.strip().startswith(('def ', 'class ', 'if ', 'for ', 'while ', 'try:', 'except:', 'finally:', 'else:', 'elif ')):
                if not line.strip().endswith(':'):
                    return 'missing_colon'
        
        # Check for indentation issues
        for i, line in enumerate(lines):
            if line.strip() and not line.startswith(' ') and i > 0:
                prev_line = lines[i-1].strip()
                if prev_line.endswith(':'):
                    return 'indentation_error'
        
        # Check for unclosed parentheses/brackets
        open_paren = code.count('(')
        close_paren = code.count(')')
        open_bracket = code.count('[')
        close_bracket = code.count(']')
        open_brace = code.count('{')
        close_brace = code.count('}')
        
        if open_paren != close_paren:
            return 'missing_paren'
        if open_bracket != close_bracket:
            return 'missing_paren'
        if open_brace != close_brace:
            return 'missing_paren'
        
        # Check for unclosed strings
        single_quotes = code.count("'")
        double_quotes = code.count('"')
        if single_quotes % 2 != 0 or double_quotes % 2 != 0:
            return 'unclosed_string'
        
        return 'valid_syntax'
    
    def _get_message(self, has_error: bool, error_type: str) -> str:
        """Generate user-friendly message"""
        if not has_error:
            return "✅ Code appears to be syntactically correct"
        else:
            messages = {
                'missing_colon': "❌ Missing colon after function/class definition",
                'indentation_error': "❌ Incorrect indentation detected",
                'missing_paren': "❌ Missing closing parenthesis/bracket",
                'unclosed_string': "❌ Unclosed string literal",
                'valid_syntax': "❌ Syntax error detected",
                'Analysis error': "❌ Error during analysis"
            }
            return messages.get(error_type, "❌ Syntax error detected")
    
    def _get_details(self, has_error: bool, error_type: str, confidence: float) -> str:
        """Generate detailed explanation"""
        if not has_error:
            return f"Model confidence: {confidence:.2%}. The code structure appears valid."
        else:
            details = {
                'missing_colon': f"Add colons (:) after function definitions, class definitions, and control statements. Confidence: {confidence:.2%}",
                'indentation_error': f"Check indentation levels. Python uses indentation to define code blocks. Confidence: {confidence:.2%}",
                'missing_paren': f"Ensure all parentheses, brackets, and braces are properly closed. Confidence: {confidence:.2%}",
                'unclosed_string': f"Check for unclosed string literals (quotes). Confidence: {confidence:.2%}",
                'valid_syntax': f"Syntax error detected with {confidence:.2%} confidence. Review the code structure.",
                'Analysis error': f"Could not analyze the code properly. Confidence: {confidence:.2%}"
            }
            return details.get(error_type, f"Syntax error detected with {confidence:.2%} confidence")

def demo_interface():
    """Interactive demo interface"""
    print("=" * 60)
    print("🔍 SYNTAX ERROR DETECTOR")
    print("=" * 60)
    print("This tool analyzes Python code for syntax errors using AI")
    print()
    
    # Initialize detector
    detector = SyntaxErrorDetector()
    
    # Demo examples
    demo_codes = [
        {
            'name': 'Valid Function',
            'code': '''def add_numbers(a, b):
    return a + b

result = add_numbers(5, 3)
print(result)'''
        },
        {
            'name': 'Missing Colon Error',
            'code': '''def add_numbers(a, b)
    return a + b

result = add_numbers(5, 3)
print(result)'''
        },
        {
            'name': 'Indentation Error',
            'code': '''def add_numbers(a, b):
    return a + b
print("This line has wrong indentation")'''
        },
        {
            'name': 'Missing Parenthesis',
            'code': '''def add_numbers(a, b:
    return a + b

result = add_numbers(5, 3
print(result)'''
        },
        {
            'name': 'Unclosed String',
            'code': '''message = "This string is not closed
print(message)'''
        }
    ]
    
    print("📋 Demo Examples:")
    for i, demo in enumerate(demo_codes, 1):
        print(f"{i}. {demo['name']}")
    print()
    
    while True:
        print("Choose an option:")
        print("1. Run demo examples")
        print("2. Enter your own code")
        print("3. Exit")
        
        choice = input("\nEnter your choice (1-3): ").strip()
        
        if choice == '1':
            print("\n" + "="*60)
            print("🏃 RUNNING DEMO EXAMPLES")
            print("="*60)
            
            for demo in demo_codes:
                print(f"\n📝 {demo['name']}")
                print("-" * 40)
                print("Code:")
                print(demo['code'])
                print("\nAnalysis:")
                
                result = detector.analyze_code(demo['code'])
                print(f"Status: {result['message']}")
                print(f"Error Type: {result['error_type']}")
                print(f"Confidence: {result['confidence']:.2%}")
                print(f"Details: {result['details']}")
                print()
        
        elif choice == '2':
            print("\n" + "="*60)
            print("✏️ ENTER YOUR CODE")
            print("="*60)
            print("Enter your Python code (press Enter twice to finish):")
            
            lines = []
            while True:
                line = input()
                if line == "" and lines and lines[-1] == "":
                    break
                lines.append(line)
            
            code = '\n'.join(lines[:-1])  # Remove the last empty line
            
            if code.strip():
                print("\n" + "="*60)
                print("🔍 ANALYSIS RESULTS")
                print("="*60)
                
                result = detector.analyze_code(code)
                
                print(f"Status: {result['message']}")
                print(f"Error Type: {result['error_type']}")
                print(f"Confidence: {result['confidence']:.2%}")
                print(f"Code Length: {result['code_length']} characters")
                print(f"AST Nodes: {result['ast_nodes']}")
                print(f"\nDetails: {result['details']}")
                
                # Save result to file
                with open('analysis_result.json', 'w') as f:
                    json.dump(result, f, indent=2)
                print(f"\n💾 Results saved to 'analysis_result.json'")
            else:
                print("❌ No code entered!")
        
        elif choice == '3':
            print("\n👋 Goodbye!")
            break
        
        else:
            print("❌ Invalid choice. Please enter 1, 2, or 3.")

def batch_analyze(codes: list) -> list:
    """Analyze multiple codes at once"""
    detector = SyntaxErrorDetector()
    results = []
    
    for i, code in enumerate(codes):
        print(f"Analyzing code {i+1}/{len(codes)}...")
        result = detector.analyze_code(code)
        result['code_index'] = i
        results.append(result)
    
    return results

if __name__ == '__main__':
    demo_interface() 