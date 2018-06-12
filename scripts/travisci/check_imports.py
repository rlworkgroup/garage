#!/usr/bin/env python3
import ast
import glob
import logging
import sys
import warnings

warnings.simplefilter("ignore", category=RuntimeWarning)
logging.basicConfig(format='%(levelname)s: %(message)s')

# Add file path prefixes here to ignore them.
IGNORE = [
    "build/",
    "dist/",
    "docs/",
    "contrib/ros",
    "contrib/",
]

# Passing flag
passed = True

# Get all Python files, except those in ignored prefixes
filenames = [f for f in glob.iglob("**/*.py", recursive=True)]
for ignored in IGNORE:
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

                # Disabling the clever exec-based tester for now
                # try:
                #     exec(line, {})
                # except TypeError:
                #     pass
                # except AttributeError:
                #     logging.error(("{filename}:{line_num} - \"{line}\" "
                #                    "failed with an AttributeError.").format(
                #                        filename=fn,
                #                        line_num=node.lineno,
                #                        line=line))
                #     passed = False
                # except SyntaxError:
                #     logging.error(("{filename}:{line_num} - \"{line}\" "
                #                    "failed with a SyntaxError.").format(
                #                        filename=fn,
                #                        line_num=node.lineno,
                #                        line=line))
                #     passed = False
                #
                # except ImportError:
                #     logging.error(
                #         "{filename}:{line_num} - \"{line}\" failed.".format(
                #             filename=fn, line_num=node.lineno, line=line))
                #     passed = False

                # Send it back to bash, instead of using exec
                print(line)

# exit code is non-zero if the script finished without error
sys.exit(not passed)
