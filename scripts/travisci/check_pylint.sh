#!/bin/bash
pylint --disable=all --enable=C0301,C0330 $(git diff HEAD HEAD~1 --name-only | grep "\.py")
