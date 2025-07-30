import os
import torch
from torch_geometric.data import DataLoader, Data
from graph_builder import ast_to_graph, NODE_TYPE_TO_IDX
from model import SyntaxGCN
from parser_util import parse_code
from sklearn.model_selection import train_test_split

DATA_DIR = 'data'
EPOCHS = 30
BATCH_SIZE = 4
LR = 0.01

# Load dataset


def load_dataset():
    samples = []
    labels = []
    for fname in os.listdir(DATA_DIR):
        if not fname.endswith('.py'):
            continue
        with open(os.path.join(DATA_DIR, fname), 'r', encoding='utf-8') as f:
            code = f.read()
        root = parse_code(code)
        if root is None:
            # Syntax error
            samples.append(None)
            labels.append(1)
        else:
            samples.append(code)
            labels.append(0)
    return samples, labels


def build_graphs(samples):
    graphs = []
    for code in samples:
        if code is None:
            # Dummy graph for syntax error (single node, no edges)
            # type idx 0, depth 0, 0 children
            x = torch.tensor([[0, 0, 0]], dtype=torch.float)
            edge_index = torch.zeros((2, 0), dtype=torch.long)
            graphs.append(Data(x=x, edge_index=edge_index))
        else:
            g = ast_to_graph(code)
            graphs.append(g)
    return graphs


def main():
    samples, labels = load_dataset()
    # Build node type dict for all valid samples
    for code in samples:
        if code is not None:
            root = parse_code(code)
            from graph_builder import build_node_type_dict
            build_node_type_dict(root)
    graphs = build_graphs(samples)
    for i, g in enumerate(graphs):
        g.y = torch.tensor([labels[i]], dtype=torch.long)
        # single graph per sample
        g.batch = torch.zeros(g.x.size(0), dtype=torch.long)
    X_train, X_test, y_train, y_test = train_test_split(
        graphs, labels, test_size=0.3, random_state=42, stratify=labels)
    train_loader = DataLoader(X_train, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(X_test, batch_size=BATCH_SIZE)
    model = SyntaxGCN(num_node_types=len(NODE_TYPE_TO_IDX))
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = torch.nn.CrossEntropyLoss()
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        for batch in train_loader:
            optimizer.zero_grad()
            out = model(batch)
            loss = criterion(out, batch.y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(
            f"Epoch {epoch+1}/{EPOCHS}, Loss: {total_loss/len(train_loader):.4f}")
    # Evaluation
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in test_loader:
            out = model(batch)
            pred = out.argmax(dim=1)
            correct += (pred == batch.y).sum().item()
            total += batch.y.size(0)
    print(f"Test Accuracy: {correct/total*100:.2f}%")


if __name__ == '__main__':
    main()
