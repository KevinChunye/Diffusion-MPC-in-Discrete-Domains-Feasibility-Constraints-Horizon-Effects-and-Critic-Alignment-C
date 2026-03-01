import os
import sys

# add the project root to sys.path so that TetrisGame and TetrisGym can be imported
# when agents are run from within the 'agent' directory or from the project root.
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if project_root not in sys.path:
    sys.path.insert(0, project_root)