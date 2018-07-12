echo "Make sure this script is running under garage conda environment"

# install dependency
if [[ "$(uname)" == "Darwin" ]]; then
	brew install cmake tinyxml assimp ffmpeg
	brew reinstall boost-python3 --without-python --with-python3 --build-from-source
	conda install qt
	export PKG_CONFIG_PATH=$(dirname $(dirname $(which python)))/lib/pkgconfig
elif [[ "$(expr substr $(uname -s) 1 5)" == "Linux" ]]; then
	sudo apt-get install -y ffmpeg pkg-config qtbase5-dev libqt5opengl5-dev libpython3.5-dev \
	  libassimp-dev libboost-python-dev libtinyxml-dev
fi

# clone roboschool and bullet
read -e -p "Please enter a path to clone roboschool and bullet3 repo [eg: /home/name/code]: " PARENT_DIR
eval PARENT_DIR=${PARENT_DIR}

ROBOSCHOOL_PATH=${PARENT_DIR}/roboschool
git clone https://github.com/openai/roboschool.git ${ROBOSCHOOL_PATH}

# build bullet for roboschool
BULLET_PATH=${PARENT_DIR}/bullet3
git clone https://github.com/olegklimov/bullet3 -b roboschool_self_collision ${BULLET_PATH}

mkdir ${BULLET_PATH}/build
cd    ${BULLET_PATH}/build
cmake -DBUILD_SHARED_LIBS=ON -DUSE_DOUBLE_PRECISION=1 -DCMAKE_INSTALL_PREFIX:PATH=${ROBOSCHOOL_PATH}/roboschool/cpp-household/bullet_local_install \
  -DBUILD_CPU_DEMOS=OFF -DBUILD_BULLET2_DEMOS=OFF -DBUILD_EXTRAS=OFF  -DBUILD_UNIT_TESTS=OFF -DBUILD_CLSOCKET=OFF -DBUILD_ENET=OFF \
  -DBUILD_OPENGL3_DEMOS=OFF ..
make -j4
make install

cd ${ROBOSCHOOL_PATH}
pip install ${ROBOSCHOOL_PATH}

echo "setup done. try command below to test roboschool:"
echo "python ${ROBOSCHOOL_PATH}/agent_zoo/demo_race2.py"