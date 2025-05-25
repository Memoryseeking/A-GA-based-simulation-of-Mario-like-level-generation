#!/usr/bin/env python3
"""
Mario Level Generator Startup Script
"""

import os
import sys

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import main program
from src.main import main

if __name__ == "__main__":
    main()
