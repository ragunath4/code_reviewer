#!/usr/bin/env python3
"""
Quick debugging tool to analyze code and show error locations
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from src.core.parser_util import parse_code
from src.core.graph_builder import ast_to_graph, build_node_type_dict

def analyze_code(code):
    """Analyze code and show error details"""
    print("="*60)
    print("CODE ANALYSIS")
    print("="*60)
    print(f"Code:\n{code}")
    print("\n" + "="*60)
    
    # Parse code
    root = parse_code(code)
    if root is None:
        print("ERROR: Failed to parse code")
        return
    
    # Find error nodes
    error_nodes = []
    def find_errors(node):
        if node.type == 'ERROR':
            start_point = getattr(node, 'start_point', (0, 0))
            end_point = getattr(node, 'end_point', (0, 0))
            error_nodes.append({
                'start_line': start_point[0] + 1,
                'start_col': start_point[1],
                'end_line': end_point[0] + 1,
                'end_col': end_point[1],
                'node': node
            })
        for child in node.children:
            find_errors(child)
    
    find_errors(root)
    
    print(f"Found {len(error_nodes)} error node(s)")
    
    for i, error in enumerate(error_nodes):
        print(f"\nError {i+1}:")
        print(f"  Location: Line {error['start_line']}, Column {error['start_col']}")
        print(f"  Span: Line {error['end_line']}, Column {error['end_col']}")
        print(f"  Lines affected: {error['end_line'] - error['start_line'] + 1}")
        
        # Show the problematic code section
        lines = code.split('\n')
        start_line = error['start_line'] - 1
        end_line = min(error['end_line'], len(lines))
        
        print(f"  Code section:")
        for j in range(start_line, end_line):
            prefix = ">>> " if j == start_line else "    "
            print(f"  {prefix}{j+1:3d}: {lines[j]}")
    
    # Convert to graph
    build_node_type_dict()
    graph = ast_to_graph(code)
    
    if graph:
        print(f"\nGraph created with {graph.num_nodes} nodes")
        
        # Find nodes with error flags
        error_graph_nodes = []
        for i in range(graph.num_nodes):
            features = graph.x[i].tolist()
            if features[3] == 1:  # error_flag
                error_graph_nodes.append({
                    'node_id': i,
                    'start_line': int(features[6]),
                    'start_col': int(features[7]),
                    'end_line': int(features[8]),
                    'end_col': int(features[9]),
                    'error_type_id': int(features[10])
                })
        
        print(f"Graph contains {len(error_graph_nodes)} error nodes")
        for error in error_graph_nodes:
            error_type = "Missing token" if error['error_type_id'] == 1 else "Unexpected token"
            print(f"  Node {error['node_id']}: Line {error['start_line']}, Col {error['start_col']} - {error_type}")

if __name__ == "__main__":
    # Test with example code
    test_code = """def hello()
    print("Hello, World!
    return True"""
    
    print("Testing with code that has syntax errors...")
    analyze_code(test_code)
    
    print("\n" + "="*60)
    print("Enter your own code to analyze:")
    print("(Press Ctrl+D or Ctrl+Z when done)")
    
    try:
        user_code = ""
        while True:
            line = input()
            user_code += line + "\n"
    except EOFError:
        if user_code.strip():
            analyze_code(user_code)
        else:
            print("No code entered.") 