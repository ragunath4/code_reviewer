#!/usr/bin/env python3
"""
Main entry point for the Syntax Error Detector
"""

import sys
import os

# Add the src directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from analysis.syntax_analyzer_gcn_only import main


if __name__ == '__main__':
    main()
