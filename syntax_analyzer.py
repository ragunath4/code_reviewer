#!/usr/bin/env python3
"""
Syntax Error Analyzer
A comprehensive tool for analyzing Python code syntax errors using Graph Convolutional Networks

Usage:
    python syntax_analyzer.py --file <filename>
    python syntax_analyzer.py --code "your code here"
"""

import argparse
import os
import sys
import torch
import json
import logging
from typing import Dict, Any, Optional, List
from pathlib import Path

# Import project modules
from parser_util import parse_code
from graph_builder import ast_to_graph, NODE_TYPE_TO_IDX, build_node_type_dict
from enhanced_model import EnhancedSyntaxGCN
from torch_geometric.data import Data

# Set up logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class SyntaxAnalyzer:
    """Comprehensive syntax error analyzer using GCN models"""

    def __init__(self, model_path: str = 'syntax_error_model.pth'):
        """Initialize the syntax analyzer"""
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")

        # Initialize model
        self.model = None
        self.model_path = model_path
        self._load_model()

        # Error type mappings
        self.error_types = {
            'missing_colon': {
                'description': 'Missing colon after function/class definition or control structure',
                'examples': ['def func()', 'if condition', 'for item in items'],
                'severity': 'high'
            },
            'indentation_error': {
                'description': 'Incorrect indentation in code blocks',
                'examples': ['def func():\nreturn value', 'if x:\nprint(x)'],
                'severity': 'high'
            },
            'missing_paren': {
                'description': 'Missing closing parenthesis, bracket, or brace',
                'examples': ['print("hello"', 'def func(a, b', 'list = [1, 2, 3'],
                'severity': 'high'
            },
            'unclosed_string': {
                'description': 'Unclosed string literal',
                'examples': ['print("hello)', 'text = "unclosed'],
                'severity': 'high'
            },
            'invalid_syntax': {
                'description': 'Invalid syntax structure',
                'examples': ['def = 5', 'if x = y:', 'import from module'],
                'severity': 'high'
            },
            'parsing_failed': {
                'description': 'Code could not be parsed by the parser',
                'examples': ['def func()\n    pass', 'if x:\nprint(x)'],
                'severity': 'critical'
            }
        }

    def _load_model(self):
        """Load the trained model"""
        try:
            if os.path.exists(self.model_path):
                # Initialize model with default node types if dict is empty
                if not NODE_TYPE_TO_IDX:
                    # Add some common node types
                    common_types = ['module', 'function_definition', 'class_definition',
                                    'if_statement', 'for_statement', 'while_statement',
                                    'expression_statement', 'assignment', 'call']
                    for i, node_type in enumerate(common_types):
                        NODE_TYPE_TO_IDX[node_type] = i

                # Ensure minimum size
                num_node_types = max(len(NODE_TYPE_TO_IDX), 50)
                self.model = EnhancedSyntaxGCN(
                    num_node_types=num_node_types).to(self.device)
                self.model.load_state_dict(torch.load(
                    self.model_path, map_location=self.device))
                logger.info(f"Loaded model from {self.model_path}")
            else:
                logger.warning(
                    f"Model not found at {self.model_path}. Using rule-based analysis only.")
                self.model = None
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            self.model = None

    def _detect_error_type(self, code: str) -> str:
        """Detect specific error types in the code"""
        lines = code.split('\n')

        # Check for missing colons
        for i, line in enumerate(lines):
            stripped = line.strip()
            if stripped.startswith(('def ', 'class ', 'if ', 'for ', 'while ', 'try:', 'except:', 'finally:', 'else:', 'elif ')):
                if not stripped.endswith(':'):
                    return 'missing_colon'

        # Check for indentation errors
        for i, line in enumerate(lines):
            if line.strip() and not line.startswith(' ') and i > 0:
                prev_line = lines[i-1].strip()
                if prev_line.endswith(':'):
                    return 'indentation_error'

        # Check for missing parentheses/brackets
        open_paren = code.count('(')
        close_paren = code.count(')')
        open_bracket = code.count('[')
        close_bracket = code.count(']')
        open_brace = code.count('{')
        close_brace = code.count('}')

        if open_paren != close_paren or open_bracket != close_bracket or open_brace != close_brace:
            return 'missing_paren'

        # Check for unclosed strings
        single_quotes = code.count("'")
        double_quotes = code.count('"')
        if single_quotes % 2 != 0 or double_quotes % 2 != 0:
            return 'unclosed_string'

        # Check for invalid variable names
        for line in lines:
            if line.strip() and '=' in line:
                var_name = line.split('=')[0].strip()
                if var_name and var_name[0].isdigit():
                    return 'invalid_syntax'

        return 'valid_syntax'

    def _rule_based_analysis(self, code: str) -> Dict[str, Any]:
        """Perform rule-based syntax analysis"""
        root = parse_code(code)

        if root is None:
            return {
                'has_syntax_error': True,
                'confidence': 0.95,
                'error_type': 'parsing_failed',
                'message': 'Code contains syntax errors that prevent parsing',
                'details': 'The code could not be parsed by Tree-sitter, indicating critical syntax errors',
                'severity': 'critical'
            }

        # Check for specific error types
        error_type = self._detect_error_type(code)
        has_error = error_type != 'valid_syntax'

        return {
            'has_syntax_error': has_error,
            'confidence': 0.85 if has_error else 0.9,
            'error_type': error_type,
            'message': f"Syntax error detected: {self.error_types.get(error_type, {}).get('description', 'Unknown error')}" if has_error else "Code is syntactically correct",
            'details': f"Error type: {error_type}, Severity: {self.error_types.get(error_type, {}).get('severity', 'unknown')}",
            'severity': self.error_types.get(error_type, {}).get('severity', 'unknown')
        }

    def _gcn_analysis(self, code: str) -> Dict[str, Any]:
        """Perform GCN-based syntax analysis"""
        try:
            # Parse code
            root = parse_code(code)

            if root is None:
                return {
                    'has_syntax_error': True,
                    'confidence': 0.98,
                    'error_type': 'parsing_failed',
                    'message': 'Code contains syntax errors that prevent parsing',
                    'details': 'The code could not be parsed by Tree-sitter, indicating critical syntax errors',
                    'severity': 'critical'
                }

            # Build node type dictionary
            build_node_type_dict(root)

            # Convert to graph
            graph = ast_to_graph(code)
            if graph is None:
                return {
                    'has_syntax_error': True,
                    'confidence': 0.9,
                    'error_type': 'parsing_failed',
                    'message': 'Failed to build AST graph',
                    'details': 'Could not convert code to graph representation',
                    'severity': 'critical'
                }

            # Prepare graph for model
            graph.batch = torch.zeros(graph.x.size(0), dtype=torch.long)
            graph = graph.to(self.device)

            # Make prediction
            self.model.eval()
            with torch.no_grad():
                output = self.model(graph)
                probabilities = torch.softmax(output, dim=1)
                prediction = output.argmax(dim=1).item()
                confidence = probabilities.max().item()

            # Interpret results
            has_error = prediction == 1
            error_type = self._detect_error_type(code)

            return {
                'has_syntax_error': has_error,
                'confidence': confidence,
                'error_type': error_type,
                'message': f"GCN prediction: {'Syntax error detected' if has_error else 'Code is syntactically correct'}",
                'details': f"Model confidence: {confidence:.2%}, Error type: {error_type}",
                'severity': self.error_types.get(error_type, {}).get('severity', 'unknown')
            }

        except Exception as e:
            logger.error(f"Error in GCN analysis: {e}")
            return self._rule_based_analysis(code)

    def analyze_code(self, code: str) -> Dict[str, Any]:
        """
        Analyze Python code for syntax errors

        Args:
            code (str): Python code to analyze

        Returns:
            dict: Comprehensive analysis results
        """
        logger.info("Starting code analysis...")

        # Perform both rule-based and GCN analysis
        rule_result = self._rule_based_analysis(code)

        if self.model is not None:
            gcn_result = self._gcn_analysis(code)

            # Combine results (give more weight to GCN if available)
            if gcn_result['confidence'] > rule_result['confidence']:
                final_result = gcn_result
                analysis_method = 'GCN + Rule-based'
            else:
                final_result = rule_result
                analysis_method = 'Rule-based + GCN'
        else:
            final_result = rule_result
            analysis_method = 'Rule-based only'

        # Add additional metadata
        final_result.update({
            'code_length': len(code),
            'lines_of_code': len(code.split('\n')),
            'analysis_method': analysis_method,
            'error_info': self.error_types.get(final_result['error_type'], {}),
            'recommendations': self._get_recommendations(final_result['error_type'])
        })

        return final_result

    def _get_recommendations(self, error_type: str) -> List[str]:
        """Get recommendations for fixing the error"""
        recommendations = {
            'missing_colon': [
                "Add a colon (:) after function/class definitions",
                "Add a colon (:) after control structures (if, for, while, etc.)",
                "Check all function and class definitions"
            ],
            'indentation_error': [
                "Ensure consistent indentation (use spaces or tabs, not both)",
                "Check that code blocks after colons are properly indented",
                "Use 4 spaces for each indentation level"
            ],
            'missing_paren': [
                "Check for matching parentheses, brackets, and braces",
                "Count opening and closing symbols",
                "Use an IDE with bracket matching"
            ],
            'unclosed_string': [
                "Check for matching quotes (single or double)",
                "Ensure all string literals are properly closed",
                "Use triple quotes for multi-line strings"
            ],
            'invalid_syntax': [
                "Review the syntax of the problematic line",
                "Check for invalid variable names (cannot start with numbers)",
                "Ensure proper Python syntax"
            ],
            'parsing_failed': [
                "Review the entire code for syntax errors",
                "Check for missing colons, parentheses, or quotes",
                "Use a Python linter for detailed error messages"
            ]
        }

        return recommendations.get(error_type, ["Review the code for syntax errors"])

    def print_analysis(self, result: Dict[str, Any], code: str = None):
        """Print formatted analysis results"""
        print("\n" + "="*60)
        print("SYNTAX ANALYSIS RESULTS")
        print("="*60)

        # Status
        status = "❌ SYNTAX ERROR" if result['has_syntax_error'] else "✅ VALID SYNTAX"
        print(f"Status: {status}")

        # Confidence
        confidence_pct = result['confidence'] * 100
        print(f"Confidence: {confidence_pct:.1f}%")

        # Error details
        if result['has_syntax_error']:
            error_info = result.get('error_info', {})
            print(
                f"\nError Type: {result['error_type'].replace('_', ' ').title()}")
            print(
                f"Description: {error_info.get('description', 'Unknown error')}")
            print(f"Severity: {error_info.get('severity', 'unknown').upper()}")

            # Recommendations
            recommendations = result.get('recommendations', [])
            if recommendations:
                print(f"\nRecommendations:")
                for i, rec in enumerate(recommendations, 1):
                    print(f"  {i}. {rec}")
        else:
            print(f"\n✅ Code is syntactically correct!")

        # Analysis method
        print(f"\nAnalysis Method: {result['analysis_method']}")
        print(f"Code Length: {result['code_length']} characters")
        print(f"Lines of Code: {result['lines_of_code']}")

        print("="*60)

    def save_analysis(self, result: Dict[str, Any], output_file: str = 'analysis_result.json'):
        """Save analysis results to JSON file"""
        try:
            with open(output_file, 'w') as f:
                json.dump(result, f, indent=2)
            logger.info(f"Analysis results saved to {output_file}")
        except Exception as e:
            logger.error(f"Error saving results: {e}")


def main():
    """Main function for command-line interface"""
    parser = argparse.ArgumentParser(
        description="Python Syntax Error Analyzer using Graph Convolutional Networks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python syntax_analyzer.py --file my_script.py
  python syntax_analyzer.py --code "def hello(): print('world')"
  python syntax_analyzer.py --file test.py --output results.json
        """
    )

    parser.add_argument('--file', type=str,
                        help='Path to Python file to analyze')
    parser.add_argument('--code', type=str,
                        help='Python code string to analyze')
    parser.add_argument('--output', type=str, default='analysis_result.json',
                        help='Output file for JSON results (default: analysis_result.json)')
    parser.add_argument('--model', type=str, default='syntax_error_model.pth',
                        help='Path to trained model file (default: syntax_error_model.pth)')
    parser.add_argument('--verbose', '-v',
                        action='store_true', help='Verbose output')

    args = parser.parse_args()

    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Validate arguments
    if not args.file and not args.code:
        parser.error("Either --file or --code must be specified")

    if args.file and args.code:
        parser.error("Cannot specify both --file and --code")

    # Initialize analyzer
    analyzer = SyntaxAnalyzer(model_path=args.model)

    try:
        # Read code
        if args.file:
            if not os.path.exists(args.file):
                print(f"Error: File '{args.file}' not found")
                sys.exit(1)

            with open(args.file, 'r', encoding='utf-8') as f:
                code = f.read()

            print(f"Analyzing file: {args.file}")
        else:
            code = args.code
            # Handle newline characters in command line input
            code = code.replace('\\n', '\n')
            print("Analyzing provided code...")

        # Perform analysis
        result = analyzer.analyze_code(code)

        # Print results
        analyzer.print_analysis(result, code)

        # Save results
        analyzer.save_analysis(result, args.output)

        # Exit with appropriate code
        sys.exit(1 if result['has_syntax_error'] else 0)

    except KeyboardInterrupt:
        print("\nAnalysis interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error during analysis: {e}")
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
