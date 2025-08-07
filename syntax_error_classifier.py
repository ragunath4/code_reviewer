#!/usr/bin/env python3
"""
Python Syntax Error Classifier using GNN
Detects and classifies syntax errors in Python code using graph neural networks.
"""

import torch
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.nn import GCNConv, global_mean_pool, global_max_pool
from torch_geometric.data import Data
import json
import re
from typing import Dict, List, Tuple
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TokenBasedGraphBuilder:
    """Builds graph representations from Python code tokens"""

    def __init__(self):
        self.token_patterns = {
            'keyword': r'\b(def|class|if|else|elif|for|while|try|except|finally|with|import|from|as|return|pass|break|continue|raise|yield|lambda|in|is|not|and|or)\b',
            'operator': r'[\+\-\*/=<>!&\|%]',
            'delimiter': r'[\(\)\[\]\{\}:,;\.]',
            'string': r'["\'](?:[^"\']|\\["\'])*["\']?',
            'number': r'\b\d+\.?\d*\b',
            'identifier': r'\b[a-zA-Z_][a-zA-Z0-9_]*\b',
            'whitespace': r'\s+',
            'comment': r'#.*$',
            'indent': r'^\s+',
        }

    def tokenize_code(self, code: str) -> List[Tuple[str, str, int]]:
        """Tokenize code into (token, type, position) tuples"""
        tokens = []
        lines = code.split('\n')

        for line_num, line in enumerate(lines):
            pos = 0
            while pos < len(line):
                matched = False

                # Check for indentation
                if pos == 0 and line.startswith(' '):
                    indent_level = len(line) - len(line.lstrip())
                    tokens.append(('INDENT', 'indent', line_num))
                    pos += indent_level
                    continue

                # Try to match each token pattern
                for token_type, pattern in self.token_patterns.items():
                    match = re.match(pattern, line[pos:])
                    if match:
                        token = match.group(0)
                        tokens.append((token, token_type, line_num))
                        pos += len(token)
                        matched = True
                        break

                if not matched:
                    # Unknown token
                    tokens.append((line[pos], 'unknown', line_num))
                    pos += 1

        return tokens

    def build_graph(self, code: str) -> Data:
        """Build graph representation from code"""
        tokens = self.tokenize_code(code)

        if not tokens:
            # Empty code - create minimal graph
            x = torch.tensor([[0, 0, 0]], dtype=torch.float)
            edge_index = torch.zeros((2, 0), dtype=torch.long)
            return Data(x=x, edge_index=edge_index)

        # Create node features
        node_features = []
        for token, token_type, line_num in tokens:
            # Feature vector: [token_type_id, line_num, token_length]
            type_id = self._get_token_type_id(token_type)
            features = [type_id, line_num, len(token)]
            node_features.append(features)

        x = torch.tensor(node_features, dtype=torch.float)

        # Create edges (sequential connections + structural connections)
        edges = []
        for i in range(len(tokens) - 1):
            # Sequential edge
            edges.append([i, i + 1])
            edges.append([i + 1, i])  # Bidirectional

        # Add structural edges based on indentation
        for i, (token, token_type, line_num) in enumerate(tokens):
            if token_type == 'indent':
                # Connect to previous non-indent token
                for j in range(i - 1, -1, -1):
                    if tokens[j][1] != 'indent':
                        edges.append([j, i])
                        edges.append([i, j])
                        break

        if edges:
            edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        else:
            edge_index = torch.zeros((2, 0), dtype=torch.long)

        return Data(x=x, edge_index=edge_index)

    def _get_token_type_id(self, token_type: str) -> int:
        """Get numeric ID for token type"""
        type_mapping = {
            'keyword': 0,
            'operator': 1,
            'delimiter': 2,
            'string': 3,
            'number': 4,
            'identifier': 5,
            'whitespace': 6,
            'comment': 7,
            'indent': 8,
            'unknown': 9
        }
        return type_mapping.get(token_type, 9)


class SyntaxErrorGNN(nn.Module):
    """Graph Neural Network for syntax error classification"""

    def __init__(self, input_dim=3, hidden_dim=64, num_classes=6):
        super().__init__()

        # Graph convolution layers
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, hidden_dim)

        # Batch normalization
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.bn3 = nn.BatchNorm1d(hidden_dim)

        # Dropout
        self.dropout = nn.Dropout(0.3)

        # Classification layers
        self.lin1 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.lin2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.lin3 = nn.Linear(hidden_dim // 2, num_classes)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # Graph convolutions
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout(x)

        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.dropout(x)

        x = self.conv3(x, edge_index)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.dropout(x)

        # Global pooling
        x_mean = global_mean_pool(x, batch)
        x_max = global_max_pool(x, batch)
        x = torch.cat([x_mean, x_max], dim=1)

        # Classification
        x = self.lin1(x)
        x = F.relu(x)
        x = self.dropout(x)

        x = self.lin2(x)
        x = F.relu(x)
        x = self.dropout(x)

        x = self.lin3(x)
        return x


class SyntaxErrorClassifier:
    """Main classifier for Python syntax errors"""

    def __init__(self, model_path: str = 'syntax_error_model.pth'):
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')
        self.graph_builder = TokenBasedGraphBuilder()
        self.model = None
        self.error_types = [
            'valid',
            'missing_colon',
            'unclosed_string',
            'unexpected_indent',
            'unexpected_eof',
            'invalid_token'
        ]

        self._load_model(model_path)

    def _load_model(self, model_path: str):
        """Load the trained model"""
        try:
            # Load the state dict
            if torch.cuda.is_available():
                state_dict = torch.load(model_path)
            else:
                state_dict = torch.load(model_path, map_location='cpu')

            # Create model
            self.model = SyntaxErrorGNN().to(self.device)

            # Load state dict
            self.model.load_state_dict(state_dict)
            self.model.eval()
            logger.info(f"Model loaded successfully from {model_path}")

        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise

    def analyze_code(self, code: str) -> Dict[str, any]:
        """Analyze Python code and return error classification"""
        if self.model is None:
            raise RuntimeError("Model not loaded")

        # Build graph and run inference
        graph = self.graph_builder.build_graph(code)
        graph = graph.to(self.device)

        with torch.no_grad():
            output = self.model(graph)
            probabilities = F.softmax(output, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0][predicted_class].item()

            result = self.error_types[predicted_class]

            return {
                "result": result,
                "confidence": confidence
            }


def main():
    """Main function for testing the classifier"""
    classifier = SyntaxErrorClassifier()

    # Test cases
    test_cases = [
        'def foo()\n  print("hi")',  # missing_colon
        'print("hello',  # unclosed_string
        'if x > 5\nprint(x)',  # missing_colon
        'x = [1, 2, 3',  # unexpected_eof
        'def test():\n    pass',  # valid
        '  print("indented")',  # unexpected_indent
        'x = @invalid',  # invalid_token
    ]

    print("Python Syntax Error Classifier Test Results:")
    print("=" * 50)

    for i, code in enumerate(test_cases, 1):
        result = classifier.analyze_code(code)
        print(f"Test {i}: {repr(code)}")
        print(
            f"Result: {result['result']} (confidence: {result['confidence']:.2f})")
        print("-" * 30)


if __name__ == "__main__":
    main()
