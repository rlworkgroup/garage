#!/bin/bash
# Make sure that conda is available

hash conda 2>/dev/null || {
  echo "Please install anaconda before continuing. You can download it at" \
    "https://www.continuum.io/downloads. Please use the Python 2.7" \
    "installer."
  exit 0
}

echo "Installing system dependencies"
echo "You will probably be asked for your sudo password."
sudo apt-get update
sudo apt-get install -y \
  python-pip \
  python-dev \
  swig cmake \
  build-essential \
  zlib1g-dev \
  libopenmpi-dev \
  ffmpeg \
  pkg-config \
  qtbase5-dev \
  libqt5opengl5-dev \
  libpython3.5-dev \
  libassimp-dev \
  libboost-python-dev \
  libtinyxml-dev
sudo apt-get build-dep -y python-pygame
sudo apt-get build-dep -y python-scipy

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
if [[ "${?}" -ne 0 ]]; then
  echo -e "\e[0;31mError: conda environment could not be created\e[0m"
  exit 1
fi

env_name="garage"
tf_version="1.8"
read -r -p "Install tensorflow-gpu instead of regular tensorflow [y/N]: " \
  response
case "${response}" in
  [yY])
    source activate "${env_name}"
    pip install tensorflow-gpu=="${tf_version}"
    source deactivate
    ;;
  *)
    source activate "${env_name}"
    pip install tensorflow=="${tf_version}"
    source deactivate
    ;;
esac

echo "Updating conda environment..."
conda env update
if [[ "${?}" -ne 0 ]]; then
  echo -e "\e[0;31mError: conda environment could not be updated\e[0m"
  exit 1
fi

echo "Setting up Roboschool..."
temp_dir="$(mktemp -d)"
roboschool_path="${temp_dir}/roboschool"
git clone --depth 1 https://github.com/openai/roboschool.git "${roboschool_path}"

# build bullet for roboschool
bullet3_path="${temp_dir}/bullet3"
bullet3_install_path="${roboschool_path}/roboschool/cpp-household/bullet_local_install"
git clone --depth 1 https://github.com/olegklimov/bullet3 -b roboschool_self_collision "${bullet3_path}"
mkdir "${bullet3_path}/build"
cd    "${bullet3_path}/build"
cmake -DBUILD_SHARED_LIBS=ON -DUSE_DOUBLE_PRECISION=1 \
      -DCMAKE_INSTALL_PREFIX:PATH="${bullet3_install_path}" \
      -DBUILD_CPU_DEMOS=OFF -DBUILD_BULLET2_DEMOS=OFF -DBUILD_EXTRAS=OFF\
      -DBUILD_UNIT_TESTS=OFF -DBUILD_CLSOCKET=OFF -DBUILD_ENET=OFF \
      -DBUILD_OPENGL3_DEMOS=OFF ..
make -j"$(nproc)"
make install

# build roboschool
pip install "${roboschool_path}"


echo "Conda environment created! Make sure to run \`source activate garage\`" \
  "whenever you open a new terminal and want to run programs under garage."
