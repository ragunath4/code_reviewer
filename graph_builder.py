import torch
from torch_geometric.data import Data
from parser_util import parse_code

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

# Extract features: node type (index), depth, number of children


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
    features.append([node_type_idx, depth, num_children])
    if parent_idx is not None:
        edge_index[0].append(parent_idx)
        edge_index[1].append(idx)
    for child in node.children:
        extract_features(child, depth+1, features,
                         node_indices, edge_index, idx)
    return features, edge_index


def ast_to_graph(code):
    root = parse_code(code)
    if root is None:
        return None  # Syntax error, can't build graph
    build_node_type_dict(root)
    features, edge_index = extract_features(root)
    x = torch.tensor(features, dtype=torch.float)
    edge_index = torch.tensor(edge_index, dtype=torch.long)
    data = Data(x=x, edge_index=edge_index)
    return data
