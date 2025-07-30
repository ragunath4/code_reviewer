# Syntax Error Detection in Python Code using Graph Neural Networks

This project detects syntax errors in Python code snippets using a graph-based machine learning model. It leverages Tree-sitter to parse code into ASTs, extracts features, and uses a Graph Convolutional Network (GCN) to classify code as valid or containing syntax errors.

## Features

- Uses [Tree-sitter](https://tree-sitter.github.io/tree-sitter/) to parse Python code into ASTs
- Extracts node features: node type, depth, number of children
- Builds AST graphs for each code sample
- Trains a GCN (PyTorch Geometric) to classify code as 'Syntax Error' or 'Valid'
- No pretrained models or embeddings (no CodeBERT/LLMs)

## Project Structure

- `train.py` — Training pipeline
- `graph_builder.py` — AST to graph conversion
- `parser_util.py` — Tree-sitter parsing utilities
- `model.py` — GCN model definition
- `data/` — Sample dataset (20 Python code samples, half with syntax errors)

## Installation

1. Install Python 3.8+
2. Install dependencies:
   ```bash
   pip install tree_sitter torch torch-geometric
   ```

## Usage

1. Place your code samples in the `data/` directory (see provided samples).
2. Run training:
   ```bash
   python train.py
   ```

## Notes

- Only manually crafted features are used (no transformers/LLMs).
- Designed for educational and research purposes.
