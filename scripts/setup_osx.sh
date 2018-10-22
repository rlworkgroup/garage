#!/usr/bin/env bash
# This script installs garage on OS X distributions.
#
# NOTICE: To keep consistency across this script, scripts/setup_linux.sh and
# docker/Dockerfile.ci, if there's any changes applied to this file, specially
# regarding the installation of dependencies, apply those same changes to the
# mentioned files.

# Exit if any error occurs
set -e

# Add OS X versions where garage is successfully installed in this list
VERIFIED_OSX_VERSIONS=(
  "10.12",
  "10.13.6",
)

### START OF CODE GENERATED BY Argbash v2.6.1 one line above ###
die()
{
  local _ret=$2
  test -n "$_ret" || _ret=1
  test "$_PRINT_HELP" = yes && print_help >&2
  echo "$1" >&2
  exit ${_ret}
}

begins_with_short_option()
{
  local first_option all_short_options
  all_short_options='h'
  first_option="${1:0:1}"
  test "$all_short_options" = "${all_short_options/$first_option/}" && \
    return 1 || return 0
}



# THE DEFAULTS INITIALIZATION - POSITIONALS
_positionals=()
# THE DEFAULTS INITIALIZATION - OPTIONALS
_arg_mjkey=
_arg_modify_bashrc="off"

print_help ()
{
  printf '%s\n' "Installer of garage for OS X."
  printf 'Usage: %s [--mjkey <arg>] [--(no-)modify-bashrc] ' "$0"
  printf '[-h|--help]\n'
  printf '\t%s\n' "--mjkey: Path of the MuJoCo key (no default)"
  printf '\t%s' "--modify-bashrc,--no-modify-bashrc: Set environment "
  printf '%s\n' "variables in .bash_profile (off by default)"
  printf '\t%s\n' "-h,--help: Prints help"
}

parse_commandline ()
{
  while test $# -gt 0
  do
    _key="$1"
    case "$_key" in
      --mjkey)
        test $# -lt 2 && \
          die "Missing value for the optional argument '$_key'." 1
        _arg_mjkey="$2"
        shift
        ;;
      --mjkey=*)
        _arg_mjkey="${_key##--mjkey=}"
        ;;
      --no-modify-bashrc|--modify-bashrc)
        _arg_modify_bashrc="on"
        test "${1:0:5}" = "--no-" && _arg_modify_bashrc="off"
        ;;
      -h|--help)
        print_help
        exit 0
        ;;
      -h*)
        print_help
        exit 0
        ;;
      *)
        _PRINT_HELP=yes die "FATAL ERROR: Got an unexpected argument '$1'" 1
        ;;
    esac
    shift
  done
}


parse_commandline "$@"
### END OF CODE GENERATED BY Argbash (sortof) ### ])

# Utility functions
script_dir_path() {
  SCRIPT_DIR="$(dirname ${0})"
  [[ "${SCRIPT_DIR}" = /* ]] && echo "${SCRIPT_DIR}" || \
    echo "${PWD}/${SCRIPT_DIR#./}"
}

# red text
print_error() {
  echo -e "\033[0;31m${@}\033[0m"
}

# yellow text
print_warning() {
  echo -e "\033[0;33m${@}\033[0m"
}

# Obtain the OS X version
VER="$(sw_vers -productVersion)"

if [[ ! " ${VERIFIED_OSX_VERSIONS[@]} " =~ " ${VER} " ]]; then
  print_warning "You are attempting to install garage on a version of OS X" \
    "which we have not verified is working." | fold -s
  print_warning "\ngarage relies on community contributions to support OS X\n"
  print_warning "If this installation is successful, please add your OS X" \
    "version to VERIFIED_OSX_VERSIONS to" \
    "https://github.com/rlworkgroup/garage/blob/master/scripts/setup_osx.sh" \
    "on GitHub and submit a pull request to rlworkgroup/garage to help out" \
    "future users. If the installation is not initially successful, but you" \
    "find changes which fix it, please help us out by submitting a PR with" \
    "your updates to the setup script." \
    | fold -s
  while [[ "${continue_var}" != "y" ]]; do
    read -p "Continue? (y/n): " continue_var
    if [[ "${continue_var}" = "n" ]]; then
      exit
    fi
  done
fi

# Verify this script is running from the correct folder (root directory)
dir_err_txt="Please run this script only from the root of the garage \
repository, i.e. you should run it using the command \
\"bash scripts/setup_osx.sh\""
if ! test -f setup.py && ! grep -Fq "name='rlgarage'," setup.py; then
  _PRINT_HELP=yes die \
  "${dir_err_txt}" 1
fi

# Verify there's a file in the mjkey path
test "$(file -b --mime-type ${_arg_mjkey})" == "text/plain" \
  || _PRINT_HELP=yes die \
  "The path ${_arg_mjkey} of the MuJoCo key is not valid." 1

# Make sure that we're under the garage directory
GARAGE_DIR="$(dirname $(script_dir_path))"
cd "${GARAGE_DIR}"

# File where environment variables are stored
BASH_PROF="${HOME}/.bash_profile"

# Install dependencies
echo "Installing garage dependencies"

# Homebrew is required first to install the other dependencies
hash brew 2>/dev/null || {
  # Install the Xcode Command Line Tools
  xcode-select --install
  # Install Homebrew
  /usr/bin/ruby -e "$(curl -fsSL \
    https://raw.githubusercontent.com/Homebrew/install/master/install)"
}

# For installing garage: bzip2, git, glfw, unzip, wget
# For building glfw: cmake
# Required for OpenAI gym: cmake boost boost-python ffmpeg sdl2 swig wget
# Required for OpenAI baselines: cmake openmpi
brew update
set +e
brew install \
  gcc@7 \
  bzip2 \
  git \
  glfw \
  unzip \
  wget \
  cmake \
  boost \
  boost-python \
  ffmpeg \
  sdl2 \
  swig \
  openmpi
set -e

# Leave a note in ~/.bash_profile for the added environment variables
if [[ "${_arg_modify_bashrc}" = on ]]; then
  echo -e "\n# Added by the garage installer" >> "${BASH_PROF}"
fi

# Set up MuJoCo
if [[ ! -d "${HOME}/.mujoco/mjpro150" ]]; then
  mkdir "${HOME}"/.mujoco
  MUJOCO_ZIP="$(mktemp -d)/mujoco.zip"
  wget https://www.roboti.us/download/mjpro150_osx.zip -O "${MUJOCO_ZIP}"
  unzip -u "${MUJOCO_ZIP}" -d "${HOME}"/.mujoco
else
  print_warning "MuJoCo is already installed"
fi
# Configure MuJoCo as a shared library
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:${HOME}/.mujoco/mjpro150/bin"
LD_LIB_ENV_VAR="LD_LIBRARY_PATH=\"\$LD_LIBRARY_PATH:${HOME}/.mujoco/mjpro150"
LD_LIB_ENV_VAR="${LD_LIB_ENV_VAR}/bin\""
if [[ "${_arg_modify_bashrc}" = on ]]; then
  echo "export ${LD_LIB_ENV_VAR}" >> "${BASH_PROF}"
fi

# Set up conda
CONDA_SH="${HOME}/miniconda2/etc/profile.d/conda.sh"
if [[ ! -d "${HOME}/miniconda2" ]]; then
  CONDA_INSTALLER="$(mktemp -d)/miniconda.sh"
  wget https://repo.continuum.io/miniconda/Miniconda2-latest-MacOSX-x86_64.sh \
    -O "${CONDA_INSTALLER}"
  chmod u+x "${CONDA_INSTALLER}"
  bash "${CONDA_INSTALLER}" -b -u
  if [[ "${_arg_modify_bashrc}" = on ]]; then
    echo ". ${CONDA_SH}" >> "${BASH_PROF}"
  fi
fi
# Export conda in this script
. "${CONDA_SH}"
conda update -q -y conda

# We need a MuJoCo key to import mujoco_py
cp "${_arg_mjkey}" "${HOME}/.mujoco/mjkey.txt"

# Create conda environment
conda env create -f environment.yml
if [[ "${?}" -ne 0 ]]; then
  print_error "Error: conda environment could not be created"
fi

# Extras
conda activate garage
{
  # Prevent pip from complaining about available upgrades
  pip install --upgrade pip

  # 'Install' garage as an editable package
  pip install -e .

  # Install git hooks
  pre-commit install -t pre-commit
  pre-commit install -t pre-push
  pre-commit install -t commit-msg

  # Install a virtualenv for the hooks
  pre-commit
}
conda deactivate

# Copy template of personal configurations so users can set them later
cp garage/config_personal_template.py garage/config_personal.py

# Add garage to python modules
if [[ "${_arg_modify_bashrc}" != on ]]; then
  echo -e "\nRemember to execute the following commands before running garage:"
  echo "${LD_LIB_ENV_VAR}"
  echo ". ${CONDA_SH}"
  echo "You may wish to edit your .bash_profile to prepend these commands."
fi

echo -e "\ngarage is installed! To make the changes take effect, work under" \
  "a new terminal. Also, make sure to run \`conda activate garage\`" \
  "whenever you open a new terminal and want to run programs under garage." \
  | fold -s
