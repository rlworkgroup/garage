#!/usr/bin/env python3
import ast
import glob
import logging
import sys
from unittest import mock
import warnings

warnings.simplefilter("ignore", category=RuntimeWarning)
logging.basicConfig(format='%(levelname)s: %(message)s')

# Add file path prefixes here to ignore them.
IGNORE_PATHS = [
    "build/",
    "dist/",
    "docs/",
    "contrib/",
    "sandbox/",
]

IGNORE_MODULES = [
    "__future__",
]

# Mock out some modules because they are slow or error-prone
MOCK_MODULES = [
    # "dm_control",
    # "dm_control.rl",
    # "dm_control.rl.control",
    # "dm_control.rl.environment",
    "ipdb",
    "matplotlib",
    "matplotlib.pyplot",
    "matplotlib.ticker",
    "mujoco_py",
    "mujoco_py.builder",
    "mujoco_py.generated",
    "mujoco_py.generated.const",
    # "tensorflow",
    # "tensorflow.python",
    # "tensorflow.python.training",
    # "theano",
    # "theano.gradient",
    # "theano.tensor",
    # "theano.tensor.extra_ops",
    # "theano.tensor.nnet",
    # "theano.tensor.signal",
    # "theano.tensor.signal.pool",
    # "theano.sandbox",
    # "theano.sandbox.rng_mrg",
]
for m in MOCK_MODULES:
    sys.modules[m] = mock.MagicMock()

# Passing flag
passed = True

# Get all Python files, except those in ignored prefixes
filenames = [f for f in glob.iglob("**/*.py", recursive=True)]
for ignored in IGNORE_PATHS:
    filenames = [f for f in filenames if not f.startswith(ignored)]

# Get all unique import statements in the repo, and their associated nodes
for fn in filenames:
    with open(fn) as f:
        syntax = f.read()
        lines = syntax.split("\n")

        try:
            module = ast.parse(syntax)
        except SyntaxError as e:
            logging.error(("{filename}:{line_num} - \"{line}\" "
                           "has a SyntaxError.").format(
                               filename=fn,
                               line_num=e.lineno,
                               line=lines[node.lineno - 1].strip()))
            passed = False

        for node in ast.walk(module):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                line = lines[node.lineno - 1].strip()

                try:
                    if not [True for i in IGNORE_MODULES if i in line]:
                        exec(line, {})
                except TypeError:
                    pass
                except AttributeError:
                    logging.error(("{filename}:{line_num} - \"{line}\" "
                                   "failed with an AttributeError.").format(
                                       filename=fn,
                                       line_num=node.lineno,
                                       line=line))
                    passed = False
                except SyntaxError:
                    logging.error(("{filename}:{line_num} - \"{line}\" "
                                   "failed with a SyntaxError.").format(
                                       filename=fn,
                                       line_num=node.lineno,
                                       line=line))
                    passed = False

                except ImportError:
                    logging.error(
                        "{filename}:{line_num} - \"{line}\" failed.".format(
                            filename=fn, line_num=node.lineno, line=line))
                    passed = False

# exit code is non-zero if the script finished without error
sys.exit(not passed)
