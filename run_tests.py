#!/usr/bin/env python3
"""
Test runner script for ScaleTorch project.
"""

import subprocess
import sys
from pathlib import Path


def run_tests():
    """Run all tests for the ScaleTorch project."""
    project_root = Path(__file__).parent.absolute()
    tests_dir = project_root / 'tests'

    if not tests_dir.exists():
        print('Tests directory not found!')
        return 1

    # Run the tests using unittest discovery
    try:
        result = subprocess.run([
            sys.executable, '-m', 'unittest', 'discover', '-s',
            str(tests_dir), '-p', 'test_*.py', '-v'
        ],
                                cwd=project_root)

        return result.returncode
    except Exception as e:
        print(f'Error running tests: {e}')
        return 1


if __name__ == '__main__':
    exit_code = run_tests()
    sys.exit(exit_code)
