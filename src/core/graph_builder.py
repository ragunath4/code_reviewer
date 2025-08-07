import torch
from torch_geometric.data import Data
from .parser_util import parse_code

# Build a mapping from node type to index for one-hot or index encoding
NODE_TYPE_TO_IDX = {}
def build_node_type_dict(root):
    global NODE_TYPE_TO_IDX
    def visit(node):
        if node.type not in NODE_TYPE_TO_IDX:
            NODE_TYPE_TO_IDX[node.type] = len(NODE_TYPE_TO_IDX)
        for child in node.children:
            visit(child)
    visit(root)

# Feature vector structure:
# [node_type_idx, depth, num_children, error_flag, start_byte, end_byte, start_line, start_col, end_line, end_col, error_type_id]
# For non-error nodes, error_flag=0 and error_type_id=-1
# For error nodes, error_type_id is a simple classification (0=unknown, 1=missing_token, 2=unexpected_token)
def classify_error_type(node):
    if len(node.children) == 0:
        return 1  # missing_token
    else:
        return 2  # unexpected_token

def extract_features(node, depth=0, features=None, node_indices=None, edge_index=None, parent_idx=None):
    if features is None:
        features = []
    if node_indices is None:
        node_indices = []
    if edge_index is None:
        edge_index = [[], []]
    idx = len(features)
    node_indices.append(idx)
    node_type_idx = NODE_TYPE_TO_IDX[node.type]
    num_children = len(node.children)
    error_flag = 1 if node.type == "ERROR" else 0
    start_line, start_col = getattr(node, 'start_point', (-1, -1))
    end_line, end_col = getattr(node, 'end_point', (-1, -1))
    if error_flag:
        start_byte = getattr(node, 'start_byte', -1)
        end_byte = getattr(node, 'end_byte', -1)
        error_type_id = classify_error_type(node)
        features.append([
            node_type_idx, depth, num_children, error_flag,
            start_byte, end_byte, start_line, start_col, end_line, end_col, error_type_id
        ])
    else:
        features.append([
            node_type_idx, depth, num_children, error_flag,
            -1, -1, -1, -1, -1, -1, -1
        ])
    if parent_idx is not None:
        edge_index[0].append(parent_idx)
        edge_index[1].append(idx)
    for child in node.children:
        extract_features(child, depth+1, features,
                         node_indices, edge_index, idx)
    return features, edge_index

def print_graph_features(features):
    print("Feature vector structure:")
    print("[node_type_idx, depth, num_children, error_flag, start_byte, end_byte, start_line, start_col, end_line, end_col, error_type_id]")
    for i, feat in enumerate(features):
        print(f"Node {i}: {feat}")

# Usage: set debug=True to print features when building a graph
def ast_to_graph(code, debug=False):
    root = parse_code(code)
    if root is None:
        return None  # Syntax error, can't build graph
    build_node_type_dict(root)
    features, edge_index = extract_features(root)
    if debug:
        print_graph_features(features)
    x = torch.tensor(features, dtype=torch.float)
    edge_index = torch.tensor(edge_index, dtype=torch.long)
    data = Data(x=x, edge_index=edge_index)
    return data
