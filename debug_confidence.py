#!/usr/bin/env python3
"""
Debug script to understand why confidence is always 70% for valid code
"""

import torch
from parser_util import parse_code
from graph_builder import ast_to_graph, NODE_TYPE_TO_IDX, build_node_type_dict
from enhanced_model import EnhancedSyntaxGCN


def debug_confidence_issue():
    """Debug the confidence issue"""
    print("🔍 DEBUGGING CONFIDENCE ISSUE")
    print("=" * 50)

    # Test code
    code = """def add(a, w):
    return a + w
"""
    print(f"Testing code:\n{code}")

    # Step 1: Parse code
    print("\n1️⃣ Parsing code...")
    root = parse_code(code)
    if root:
        print("✅ Parser: Code is valid")
    else:
        print("❌ Parser: Code has syntax errors")
        return

    # Step 2: Build node type dict
    print("\n2️⃣ Building node type dictionary...")
    build_node_type_dict(root)
    print(f"Node types found: {len(NODE_TYPE_TO_IDX)}")

    # Step 3: Convert to graph
    print("\n3️⃣ Converting to graph...")
    graph = ast_to_graph(code)
    if graph:
        print(
            f"✅ Graph created: {graph.x.shape[0]} nodes, {graph.edge_index.shape[1]} edges")
    else:
        print("❌ Failed to create graph")
        return

    # Step 4: Prepare for model
    print("\n4️⃣ Preparing graph for model...")
    graph.batch = torch.zeros(graph.x.size(0), dtype=torch.long)

    # Step 5: Initialize model with same settings as analyzer
    print("\n5️⃣ Initializing model...")
    num_node_types = 99  # Same as analyzer
    model = EnhancedSyntaxGCN(num_node_types=num_node_types)
    print(f"Model initialized with {num_node_types} node types")

    # Step 6: Make prediction
    print("\n6️⃣ Making prediction...")
    model.eval()
    with torch.no_grad():
        try:
            output = model(graph)
            print(f"Raw output shape: {output.shape}")
            print(f"Raw output values: {output}")

            probabilities = torch.softmax(output, dim=1)
            print(f"Probabilities: {probabilities}")

            prediction = output.argmax(dim=1).item()
            confidence = probabilities.max().item()

            print(f"\n📊 PREDICTION RESULTS:")
            print(f"Prediction: {prediction} (0=Valid, 1=Invalid)")
            print(f"Confidence: {confidence:.3f}")
            print(
                f"Interpretation: {'Valid' if prediction == 0 else 'Invalid'}")

            # Simulate the analyzer logic
            print(f"\n🔍 ANALYZER LOGIC SIMULATION:")
            has_error = prediction == 1
            print(f"GCN says has_error: {has_error}")
            print(f"Parser says valid: {root is not None}")
            print(f"GCN confidence: {confidence:.3f}")

            if root is not None and has_error and confidence < 0.6:
                print("✅ Parser override triggered!")
                print("Reason: Parser says valid, GCN says invalid with low confidence")
                final_confidence = 0.7
                final_has_error = False
            else:
                print("❌ No parser override")
                final_confidence = confidence
                final_has_error = has_error

            print(f"\n🎯 FINAL RESULT:")
            print(f"Final has_error: {final_has_error}")
            print(f"Final confidence: {final_confidence:.3f}")

        except Exception as e:
            print(f"❌ Error during prediction: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    debug_confidence_issue()
