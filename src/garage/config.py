"""Config garage project path."""
import pathlib
import sys

GARAGE_PROJECT_PATH = str(pathlib.Path(__file__).parent.parent.parent)
sys.path.append(GARAGE_PROJECT_PATH)
