#!/bin/bash
# Make sure that pip is available
hash brew 2>/dev/null || {
  echo "Please install homebrew before continuing. You can use the" \
    "following command to install:"
  echo "/usr/bin/ruby -e \"\$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install)\""
  exit 0
}

hash conda 2>/dev/null || {
  echo "Please install anaconda before continuing. You can download" \
    "it at https://www.continuum.io/downloads. Please use the Python 2.7" \
    " installer."
  exit 0
}


echo "Installing system dependencies"
echo "You will probably be asked for your sudo password."

brew install swig sdl sdl_image sdl_mixer sdl_ttf portmidi openmpi

# Make sure that we're under the directory of the project
cd "$(dirname "$0")/.."

echo "Setting up MuJoCo..."
/bin/bash ./scripts/setup_mujoco.sh
if [[ "${?}" -ne 0 ]]; then
  echo -e "\e[0;31mError: MuJoCo couldn't be set up\e[0m"
  exit 1
fi

echo "Creating conda environment..."
conda env create -f environment.yml
conda env update

echo "Conda environment created! Make sure to run \`source activate garage\`" \
  "whenever you open a new terminal and want to run programs under garage."
