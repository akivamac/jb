#!/usr/bin/env python3
"""Wrapper to run generate_reviews_v2.py for all experts."""
import sys
sys.path.insert(0, '.')

import generate_reviews_v2  # This will use EMPLOYEE_TYPE or whatever the model uses internally
