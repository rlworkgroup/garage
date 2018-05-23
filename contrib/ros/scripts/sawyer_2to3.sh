#!/usr/bin/env bash
packages=(intera_sdk sawyer_simulator)
ros_ws=<YOUR_FAVORITE_WORKSPACE_PATH>/src/
for package in ${packages[@]}; do
    python_scripts=`find "${ros_ws}${package}" -name "*.py"`
    for script in ${python_scripts[@]}; do
        2to3 -w ${script}
    done
done
