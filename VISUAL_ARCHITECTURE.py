#!/usr/bin/env python3
"""
Visual Architecture Diagram Generator
Shows how all files in the syntax error detection project are connected
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, ConnectionPatch
import numpy as np


def create_architecture_diagram():
    """Create a visual diagram of the project architecture"""

    fig, ax = plt.subplots(1, 1, figsize=(20, 16))
    ax.set_xlim(0, 20)
    ax.set_ylim(0, 16)
    ax.set_aspect('equal')

    # Colors for different categories
    colors = {
        'core': '#FF6B6B',      # Red for core files
        'enhanced': '#4ECDC4',   # Teal for enhanced files
        'data': '#45B7D1',       # Blue for data
        'analysis': '#96CEB4',   # Green for analysis
        'parser': '#FFEAA7'      # Yellow for parser
    }

    # Define file positions and properties
    files = {
        # Core Files (Original)
        'model.py': {'pos': (2, 14), 'color': colors['core'], 'category': 'Core'},
        'graph_builder.py': {'pos': (2, 12), 'color': colors['core'], 'category': 'Core'},
        'parser_util.py': {'pos': (2, 10), 'color': colors['core'], 'category': 'Core'},
        'train.py': {'pos': (2, 8), 'color': colors['core'], 'category': 'Core'},

        # Enhanced Files (New)
        'enhanced_model.py': {'pos': (6, 14), 'color': colors['enhanced'], 'category': 'Enhanced'},
        'improved_trainer.py': {'pos': (6, 12), 'color': colors['enhanced'], 'category': 'Enhanced'},
        'expanded_dataset.py': {'pos': (6, 10), 'color': colors['enhanced'], 'category': 'Enhanced'},
        'model_comparison.py': {'pos': (6, 8), 'color': colors['enhanced'], 'category': 'Enhanced'},

        # Data
        'data/': {'pos': (10, 14), 'color': colors['data'], 'category': 'Data'},
        'tree-sitter-python/': {'pos': (10, 12), 'color': colors['parser'], 'category': 'Parser'},

        # Analysis Files
        'error_analysis.py': {'pos': (14, 14), 'color': colors['analysis'], 'category': 'Analysis'},
        'comprehensive_test.py': {'pos': (14, 12), 'color': colors['analysis'], 'category': 'Analysis'},

        # Output Files
        'syntax_error_model.pth': {'pos': (10, 8), 'color': '#FFA07A', 'category': 'Output'},
        'model_comparison_results.json': {'pos': (14, 8), 'color': '#FFA07A', 'category': 'Output'},
        'model_comparison.png': {'pos': (14, 6), 'color': '#FFA07A', 'category': 'Output'}
    }

    # Draw file boxes
    for filename, props in files.items():
        x, y = props['pos']
        color = props['color']
        category = props['category']

        # Create rounded rectangle
        box = FancyBboxPatch(
            (x-0.8, y-0.6), 1.6, 1.2,
            boxstyle="round,pad=0.1",
            facecolor=color,
            edgecolor='black',
            linewidth=2,
            alpha=0.8
        )
        ax.add_patch(box)

        # Add text
        ax.text(x, y, filename, ha='center', va='center',
                fontsize=8, fontweight='bold', wrap=True)

    # Draw connections
    connections = [
        # Data flow
        ('expanded_dataset.py', 'data/', 'creates'),
        ('data/', 'improved_trainer.py', 'read by'),
        ('parser_util.py', 'tree-sitter-python/', 'uses'),
        ('graph_builder.py', 'parser_util.py', 'uses'),
        ('enhanced_model.py', 'graph_builder.py', 'uses'),
        ('improved_trainer.py', 'enhanced_model.py', 'trains'),
        ('improved_trainer.py', 'syntax_error_model.pth', 'saves'),

        # Model comparison
        ('model_comparison.py', 'enhanced_model.py', 'imports'),
        ('model_comparison.py', 'data/', 'uses'),
        ('model_comparison.py', 'model_comparison_results.json', 'saves'),
        ('model_comparison.py', 'model_comparison.png', 'generates'),

        # Analysis
        ('error_analysis.py', 'model.py', 'uses'),
        ('error_analysis.py', 'graph_builder.py', 'uses'),
        ('comprehensive_test.py', 'enhanced_model.py', 'uses'),

        # Legacy connections
        ('train.py', 'model.py', 'uses'),
        ('train.py', 'graph_builder.py', 'uses'),
        ('train.py', 'parser_util.py', 'uses'),
    ]

    # Draw connection arrows
    for start, end, label in connections:
        if start in files and end in files:
            start_pos = files[start]['pos']
            end_pos = files[end]['pos']

            # Calculate arrow position
            dx = end_pos[0] - start_pos[0]
            dy = end_pos[1] - start_pos[1]

            # Normalize for arrow positioning
            length = np.sqrt(dx**2 + dy**2)
            if length > 0:
                dx_norm = dx / length
                dy_norm = dy / length

                # Start and end points for arrow
                arrow_start = (start_pos[0] + 0.8 *
                               dx_norm, start_pos[1] + 0.6 * dy_norm)
                arrow_end = (end_pos[0] - 0.8 * dx_norm,
                             end_pos[1] - 0.6 * dy_norm)

                # Draw arrow
                arrow = ConnectionPatch(
                    arrow_start, arrow_end, "data", "data",
                    arrowstyle="->", shrinkA=5, shrinkB=5,
                    mutation_scale=20, fc="black", ec="black", linewidth=2
                )
                ax.add_patch(arrow)

                # Add label
                mid_x = (arrow_start[0] + arrow_end[0]) / 2
                mid_y = (arrow_start[1] + arrow_end[1]) / 2
                ax.text(mid_x, mid_y, label, ha='center', va='center',
                        fontsize=6, bbox=dict(boxstyle="round,pad=0.2", facecolor='white', alpha=0.8))

    # Add category labels
    categories = {
        'Core Files': (1, 15.5),
        'Enhanced Files': (5, 15.5),
        'Data & Parser': (9, 15.5),
        'Analysis & Output': (13, 15.5)
    }

    for category, pos in categories.items():
        ax.text(pos[0], pos[1], category, ha='center', va='center',
                fontsize=12, fontweight='bold', bbox=dict(boxstyle="round,pad=0.5",
                                                          facecolor='lightgray', alpha=0.8))

    # Add data flow description
    flow_text = """
    DATA FLOW:
    1. expanded_dataset.py creates code samples in data/
    2. improved_trainer.py reads samples and uses parser_util.py
    3. parser_util.py uses tree-sitter-python/ to parse code
    4. graph_builder.py converts AST to graphs
    5. enhanced_model.py processes graphs for classification
    6. improved_trainer.py trains models and saves results
    7. model_comparison.py compares different architectures
    8. Analysis files test and evaluate the system
    """

    ax.text(10, 2, flow_text, ha='center', va='top', fontsize=10,
            bbox=dict(boxstyle="round,pad=0.5", facecolor='lightblue', alpha=0.8))

    # Add title
    ax.text(10, 16.5, 'Syntax Error Detection Project - Complete Architecture',
            ha='center', va='center', fontsize=16, fontweight='bold')

    # Remove axes
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)

    plt.tight_layout()
    plt.savefig('project_architecture.png', dpi=300, bbox_inches='tight')
    plt.show()


def create_simplified_flow():
    """Create a simplified data flow diagram"""

    fig, ax = plt.subplots(1, 1, figsize=(16, 8))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 8)

    # Define flow steps
    steps = [
        ('Python Code', (1, 4), '#FF6B6B'),
        ('Tree-sitter Parser', (3, 4), '#4ECDC4'),
        ('AST Tree', (5, 4), '#45B7D1'),
        ('Graph Builder', (7, 4), '#96CEB4'),
        ('Graph Data', (9, 4), '#FFEAA7'),
        ('GCN Model', (11, 4), '#FFA07A'),
        ('Prediction', (13, 4), '#98D8C8')
    ]

    # Draw flow boxes
    for step_name, pos, color in steps:
        x, y = pos

        # Create box
        box = FancyBboxPatch(
            (x-1, y-0.5), 2, 1,
            boxstyle="round,pad=0.1",
            facecolor=color,
            edgecolor='black',
            linewidth=2,
            alpha=0.8
        )
        ax.add_patch(box)

        # Add text
        ax.text(x, y, step_name, ha='center', va='center',
                fontsize=10, fontweight='bold')

    # Draw arrows
    for i in range(len(steps) - 1):
        start_pos = steps[i][1]
        end_pos = steps[i + 1][1]

        arrow = ConnectionPatch(
            (start_pos[0] + 1, start_pos[1]), (end_pos[0] - 1, end_pos[1]),
            "data", "data",
            arrowstyle="->", shrinkA=5, shrinkB=5,
            mutation_scale=20, fc="black", ec="black", linewidth=2
        )
        ax.add_patch(arrow)

    # Add labels
    labels = [
        ('Input', (1, 5.5)),
        ('Parse', (3, 5.5)),
        ('Structure', (5, 5.5)),
        ('Convert', (7, 5.5)),
        ('Features', (9, 5.5)),
        ('Process', (11, 5.5)),
        ('Output', (13, 5.5))
    ]

    for label, pos in labels:
        ax.text(pos[0], pos[1], label, ha='center', va='center',
                fontsize=8, fontweight='bold', color='darkblue')

    # Add file mappings
    file_mappings = [
        ('parser_util.py', (3, 2.5)),
        ('graph_builder.py', (7, 2.5)),
        ('enhanced_model.py', (11, 2.5))
    ]

    for file_name, pos in file_mappings:
        ax.text(pos[0], pos[1], file_name, ha='center', va='center',
                fontsize=8, bbox=dict(boxstyle="round,pad=0.2",
                                      facecolor='lightgray', alpha=0.8))

    # Add title
    ax.text(8, 7.5, 'Simplified Data Flow: Code → AST → Graph → Prediction',
            ha='center', va='center', fontsize=14, fontweight='bold')

    # Remove axes
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)

    plt.tight_layout()
    plt.savefig('simplified_flow.png', dpi=300, bbox_inches='tight')
    plt.show()


if __name__ == '__main__':
    print("Generating project architecture diagrams...")
    create_architecture_diagram()
    create_simplified_flow()
    print("Diagrams saved as 'project_architecture.png' and 'simplified_flow.png'")
