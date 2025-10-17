#!/usr/bin/env python3
"""
Simple test runner for pipeline feature tests.
Usage: python run_pipeline_tests.py
"""

import sys
import subprocess
from pathlib import Path

def run_tests():
    """Run the pipeline feature tests."""
    test_file = Path(__file__).parent / "tests" / "test_pipeline_features.py"
    
    if not test_file.exists():
        print(f"Test file not found: {test_file}")
        return False
    
    print("Running tnclean pipeline feature tests...")
    print("=" * 50)
    
    try:
        # Run pytest with verbose output
        result = subprocess.run([
            sys.executable, "-m", "pytest", 
            str(test_file), 
            "-v", 
            "--tb=short",
            "--color=yes"
        ], capture_output=False, text=True)
        
        return result.returncode == 0
        
    except FileNotFoundError:
        print("pytest not found. Please install with: pip install pytest")
        return False
    except Exception as e:
        print(f"Error running tests: {e}")
        return False

if __name__ == "__main__":
    success = run_tests()
    if success:
        print("\n✅ All pipeline tests passed!")
        sys.exit(0)
    else:
        print("\n❌ Some pipeline tests failed!")
        sys.exit(1)