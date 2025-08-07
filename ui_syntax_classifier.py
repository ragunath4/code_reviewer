import streamlit as st
import torch
from src.models.enhanced_model import UnifiedSyntaxGCN
from train_syntax_classifier import analyze_code_with_details
import json

def ast_to_text_tree(ast, indent=0, max_depth=8):
    if ast is None or indent > max_depth:
        return ""
    s = "  " * indent + f"- {ast['type']} (Line {ast['start_line']}, Col {ast['start_col']})\n"
    if isinstance(ast.get('children'), list):
        for child in ast['children']:
            s += ast_to_text_tree(child, indent+1, max_depth)
    elif ast.get('children') == '...':
        s += "  " * (indent+1) + "...\n"
    return s

st.set_page_config(page_title="Python Syntax Error Classifier", layout="centered")
st.title("🧠 Python Syntax Error Classifier (GNN)")
st.write("Paste your Python code below and click Analyze. The model will classify it as valid or invalid, and if invalid, will predict the error type. You will also see AST and graph details.")

# Load model (cache for performance)
@st.cache_resource
def load_model():
    model = UnifiedSyntaxGCN(num_node_types=100, num_error_types=6)
    model.load_state_dict(torch.load('unified_syntax_error_model.pth', map_location='cpu'))
    model.eval()
    return model

model = load_model()

code = st.text_area("Paste your Python code here:", height=200)

if st.button("Analyze"):
    if not code.strip():
        st.warning("Please enter some Python code.")
    else:
        with st.spinner("Analyzing code..."):
            result = analyze_code_with_details(model, code)
        st.markdown(f"**Result:** :blue[{result['validity'].capitalize()}]")
        if result['validity'] == 'invalid':
            st.markdown(f"**Predicted Error Type:** :red[{result['error_type'].replace('_', ' ').capitalize()}] (class {result['error_type_id']})")
            # Show error node locations
            if result['error_nodes']:
                st.markdown("**Error Node Locations (AST):**")
                for i, err in enumerate(result['error_nodes']):
                    st.markdown(f"- Error {i+1}: Line {err['start_line']}, Col {err['start_col']} to Line {err['end_line']}, Col {err['end_col']}")
            else:
                st.info("No explicit error nodes found in AST.")
        else:
            st.success("No syntax error detected!")
        # Show AST structure (tree view)
        st.markdown("---")
        st.markdown("### AST Structure (tree view, truncated)")
        if result['ast']:
            st.code(ast_to_text_tree(result['ast'], max_depth=6), language='text')
        else:
            st.info("No AST generated.")
        # Show graph node features
        st.markdown("---")
        st.markdown("### Graph Node Features (first 20 nodes)")
        if result['graph_features']:
            import pandas as pd
            df = pd.DataFrame(result['graph_features'][:20])
            # Add a column for error node location (if error_flag)
            df['error_location'] = df.apply(lambda row: f"Line {row['start_line']}, Col {row['start_col']}" if row['error_flag'] == 1 else '', axis=1)
            def highlight_error(row):
                color = 'background-color: #ffcccc' if row['error_flag'] == 1 else ''
                return [color] * len(row)
            st.dataframe(df.style.apply(highlight_error, axis=1))
        else:
            st.info("No graph features available.")
        # Show interactive graph visualization
        st.markdown("---")
        st.markdown("### Graph Visualization (pyvis)")
        try:
            from pyvis.network import Network
            import tempfile
            import os
            import pandas as pd
            if result['graph_features']:
                # Build pyvis network
                net = Network(height='400px', width='100%', directed=True, notebook=False)
                df = pd.DataFrame(result['graph_features'])
                # Add nodes
                for i, row in df.iterrows():
                    label = f"{row['node_id']}: type {row['node_type_idx']}\nLine {row['start_line']}"
                    color = '#ffcccc' if row['error_flag'] == 1 else '#97c2fc'
                    net.add_node(row['node_id'], label=label, color=color)
                # Add edges (parent->child)
                # For this, we need to reconstruct edges from the DataFrame
                # We'll use the parent-child relationship from the graph features (depth info is available, but not parent index)
                # Instead, let's use a simple heuristic: consecutive nodes with increasing depth are children
                prev_depths = []
                for i, row in df.iterrows():
                    if i == 0:
                        prev_depths = [0]
                        continue
                    curr_depth = row['depth']
                    # Find the most recent node with depth == curr_depth-1
                    parent_idx = None
                    for j in range(i-1, -1, -1):
                        if df.iloc[j]['depth'] == curr_depth - 1:
                            parent_idx = df.iloc[j]['node_id']
                            break
                    if parent_idx is not None:
                        net.add_edge(parent_idx, row['node_id'])
                # Save and display
                with tempfile.NamedTemporaryFile(delete=False, suffix='.html') as tmp_file:
                    net.show(tmp_file.name)
                    html_content = open(tmp_file.name, 'r', encoding='utf-8').read()
                st.components.v1.html(html_content, height=450, scrolling=True)
                os.unlink(tmp_file.name)
            else:
                st.info("No graph to visualize.")
        except ImportError:
            st.warning("pyvis is not installed. Run 'pip install pyvis' to enable graph visualization.")
        except Exception as e:
            st.error(f"Graph visualization error: {e}")