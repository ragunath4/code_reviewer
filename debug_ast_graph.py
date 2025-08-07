#!/usr/bin/env python3
"""
Comprehensive debugging tool for AST generation, graph construction, and error analysis.
Shows detailed information about how code is parsed, converted to AST, and transformed into graphs.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from tree_sitter import Language, Parser
from src.core.parser_util import parse_code
from src.core.graph_builder import ast_to_graph, build_node_type_dict, NODE_TYPE_TO_IDX
import torch
from src.models.enhanced_model import UnifiedSyntaxGCN

def print_ast_details(node, depth=0, max_depth=10):
    """Print detailed AST node information"""
    if depth > max_depth:
        return
    
    indent = "  " * depth
    node_type = node.type
    start_point = getattr(node, 'start_point', (0, 0))
    end_point = getattr(node, 'end_point', (0, 0))
    start_byte = getattr(node, 'start_byte', -1)
    end_byte = getattr(node, 'end_byte', -1)
    
    # Get text content if it's a leaf node
    text_content = node.text.decode('utf-8') if hasattr(node, 'text') and node.text else ""
    
    print(f"{indent}Node: {node_type}")
    print(f"{indent}  Position: Line {start_point[0]+1}, Col {start_point[1]} to Line {end_point[0]+1}, Col {end_point[1]}")
    print(f"{indent}  Bytes: {start_byte} to {end_byte}")
    if text_content:
        print(f"{indent}  Text: '{text_content}'")
    
    # Check if this is an error node
    if node_type == 'ERROR':
        print(f"{indent}  *** ERROR NODE DETECTED ***")
        print(f"{indent}  Error location: Line {start_point[0]+1}, Column {start_point[1]}")
        print(f"{indent}  Error span: {end_point[0] - start_point[0]} lines, {end_point[1] - start_point[1]} columns")
    
    # Print children
    for child in node.children:
        print_ast_details(child, depth + 1, max_depth)

def print_graph_details(graph, code):
    """Print detailed graph information"""
    print("\n" + "="*60)
    print("GRAPH DETAILS")
    print("="*60)
    
    print(f"Number of nodes: {graph.num_nodes}")
    print(f"Number of edges: {graph.num_edges}")
    print(f"Node features shape: {graph.x.shape}")
    print(f"Edge index shape: {graph.edge_index.shape}")
    
    # Print feature vector structure
    print(f"\nFeature vector structure (11 features per node):")
    print("[node_type_idx, depth, num_children, error_flag, start_byte, end_byte, start_line, start_col, end_line, end_col, error_type_id]")
    
    # Print detailed node information
    print(f"\nDetailed node information:")
    for i in range(min(graph.num_nodes, 20)):  # Show first 20 nodes
        features = graph.x[i].tolist()
        print(f"Node {i}: {features}")
        
        # Decode the features
        node_type_idx = int(features[0])
        depth = int(features[1])
        num_children = int(features[2])
        error_flag = int(features[3])
        start_byte = int(features[4])
        end_byte = int(features[5])
        start_line = int(features[6])
        start_col = int(features[7])
        end_line = int(features[8])
        end_col = int(features[9])
        error_type_id = int(features[10])
        
        print(f"  Decoded: type_idx={node_type_idx}, depth={depth}, children={num_children}")
        print(f"  Error flag: {error_flag}")
        if error_flag:
            print(f"  Error location: Line {start_line}, Col {start_col} to Line {end_line}, Col {end_col}")
            print(f"  Error type ID: {error_type_id}")
            if error_type_id == 1:
                print(f"  Error type: Missing token")
            elif error_type_id == 2:
                print(f"  Error type: Unexpected token")
    
    if graph.num_nodes > 20:
        print(f"... and {graph.num_nodes - 20} more nodes")

def analyze_code_with_details(code):
    """Analyze code with comprehensive details"""
    print("="*80)
    print("COMPREHENSIVE CODE ANALYSIS")
    print("="*80)
    print(f"Input code:\n{code}")
    print("\n" + "="*80)
    
    # Step 1: Parse code to AST
    print("STEP 1: PARSING CODE TO AST")
    print("="*40)
    root = parse_code(code)
    
    if root is None:
        print("ERROR: Failed to parse code - no AST generated")
        return None
    
    print(f"AST root node type: {root.type}")
    print(f"AST root position: {getattr(root, 'start_point', (0,0))} to {getattr(root, 'end_point', (0,0))}")
    
    # Check for syntax errors in AST
    has_errors = False
    def check_for_errors(node):
        nonlocal has_errors
        if node.type == 'ERROR':
            has_errors = True
        for child in node.children:
            check_for_errors(child)
    
    check_for_errors(root)
    print(f"AST contains syntax errors: {has_errors}")
    
    # Print detailed AST structure
    print("\nDetailed AST structure:")
    print_ast_details(root, max_depth=8)
    
    # Step 2: Convert AST to graph
    print("\n" + "="*40)
    print("STEP 2: CONVERTING AST TO GRAPH")
    print("="*40)
    
    # Initialize node type dictionary
    build_node_type_dict()
    
    # Convert to graph with debug info
    graph = ast_to_graph(code, debug=True)
    
    if graph is None:
        print("ERROR: Failed to convert AST to graph")
        return None
    
    # Print graph details
    print_graph_details(graph, code)
    
    # Step 3: Model prediction (if model available)
    print("\n" + "="*40)
    print("STEP 3: MODEL PREDICTION")
    print("="*40)
    
    model_path = 'unified_syntax_error_model.pth'
    if os.path.exists(model_path):
        try:
            model = UnifiedSyntaxGCN(num_node_types=len(NODE_TYPE_TO_IDX), num_error_types=6)
            model.load_state_dict(torch.load(model_path, map_location='cpu'))
            model.eval()
            
            with torch.no_grad():
                validity_logits, error_type_logits = model(graph)
                pred_validity = validity_logits.argmax(dim=1).item()
                pred_error_type = error_type_logits.argmax(dim=1).item()
            
            error_types = ['valid', 'missing_colon', 'unclosed_string', 'unexpected_indent', 'unexpected_eof', 'invalid_token']
            
            print(f"Model prediction:")
            print(f"  Validity: {'valid' if pred_validity == 0 else 'invalid'}")
            print(f"  Error type: {error_types[pred_error_type] if pred_error_type < len(error_types) else 'unknown'}")
            print(f"  Validity confidence: {torch.softmax(validity_logits, dim=1).max().item():.3f}")
            print(f"  Error type confidence: {torch.softmax(error_type_logits, dim=1).max().item():.3f}")
            
        except Exception as e:
            print(f"Error loading model: {e}")
    else:
        print(f"Model file not found at {model_path}")
    
    return graph

def main():
    """Main function for interactive debugging"""
    print("Python Code AST/Graph Debugger")
    print("="*50)
    
    # Example codes to test
    test_codes = [
        # Valid code
        """def hello():
    print("Hello, World!")
    return True""",
        
        # Missing colon
        """def hello()
    print("Hello, World!")""",
        
        # Unclosed string
        """def hello():
    print("Hello, World!
    return True""",
        
        # Unexpected indent
        """def hello():
print("Hello, World!")
    return True""",
        
        # Unexpected EOF
        """def hello():
    print("Hello, World!")
    return""",
        
        # Invalid token
        """def hello():
    print("Hello, World!")
    return @""",
    ]
    
    print("Available test cases:")
    for i, code in enumerate(test_codes):
        print(f"{i+1}. {code.split()[0]}...")
    
    try:
        choice = input("\nEnter test case number (1-6) or paste your own code: ")
        
        if choice.isdigit() and 1 <= int(choice) <= len(test_codes):
            code = test_codes[int(choice) - 1]
        else:
            code = choice
        
        analyze_code_with_details(code)
        
    except KeyboardInterrupt:
        print("\nExiting...")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main() 