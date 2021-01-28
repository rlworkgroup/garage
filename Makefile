SHELL := /bin/bash

.PHONY: help test build-test check docs ci-job-precommit ci-job-normal \
	ci-job-large ci-job-mujoco ci-job-mujoco-long ci-job-nightly ci-job-verify-envs \
	ci-verify-envs-conda ci-verify-envs-pipenv ci-deploy-docker build-ci build-dev \
	build-dev-nvidia run-ci run-dev run-dev-nvidia run-dev-nvidia-headless \
	assert-docker assert-travis ensure-data-path-exists assert-docker-version

.DEFAULT_GOAL := help

# Path in host where the experiment data obtained in the container is stored
DATA_PATH ?= $(shell pwd)/data
# Set the environment variable MJKEY with the contents of the file specified by
# MJKEY_PATH.
MJKEY_PATH ?= ${HOME}/.mujoco/mjkey.txt

export DOCKER_BUILDKIT=1

build-test: TAG ?= rlworkgroup/garage-test
build-test: assert-docker-version docker/Dockerfile
	docker build \
		-f docker/Dockerfile \
		--cache-from rlworkgroup/garage-test:latest \
		--cache-from rlworkgroup/garage:latest \
		--target garage-test \
		-t ${TAG} \
		${BUILD_ARGS} \
		.

test:  ## Run the garage-test docker target that runs all tests except huge and flaky
test: RUN_ARGS = --memory 7500m --memory-swap 7500m
test: TAG ?= rlworkgroup/garage-test
test: CONTAINER_NAME ?= ''
test: build-test
	@echo "Running test suite..."
	docker run \
		-it \
		--rm \
		-e MJKEY="$$(cat $(MJKEY_PATH))" \
		--name ${CONTAINER_NAME} \
		${RUN_ARGS} \
		${TAG}

docs:  ## Build HTML documentation
docs:
	@pushd docs && make html && popd
	@python -c 'import os, webbrowser; webbrowser.open("file://" + os.path.realpath("docs/_build/html/index.html"))'

ci-job-precommit: assert-docker
	scripts/travisci/check_precommit.sh
	@pushd docs && make doctest clean && popd

ci-job-normal: assert-docker
	[ ! -f $(MJKEY_PATH) ] || mv $(MJKEY_PATH) $(MJKEY_PATH).bak
	pytest --cov=garage --cov-report=xml --reruns 1 -m \
	    'not nightly and not huge and not flaky and not large and not mujoco and not mujoco_long' --durations=20
	for i in {1..5}; do \
		bash <(curl -s https://codecov.io/bash --retry 5) -Z && break \
			|| echo 'Retrying...' && sleep 30 && continue; \
		exit 1; \
	done

# Need to be able to access $!, a special bash variable
define LARGE_TEST
  pytest --cov=garage --cov-report=xml --reruns 1 -m 'large and not flaky' --durations=20 &
  PYTEST_PID=$$!
  while ps -p $$PYTEST_PID > /dev/null ; do
    echo 'Still running'
    sleep 60
  done
endef
export LARGE_TEST

ci-job-large: assert-docker
	[ ! -f $(MJKEY_PATH) ] || mv $(MJKEY_PATH) $(MJKEY_PATH).bak
	bash -c "$$LARGE_TEST"
	for i in {1..5}; do \
		bash <(curl -s https://codecov.io/bash --retry 5) -Z && break \
			|| echo 'Retrying...' && sleep 30 && continue; \
		exit 1; \
	done

ci-job-mujoco: assert-docker
	pytest --cov=garage --cov-report=xml --reruns 1 -m 'mujoco and not flaky' --durations=20
	for i in {1..5}; do \
		bash <(curl -s https://codecov.io/bash --retry 5) -Z && break \
			|| echo 'Retrying...' && sleep 30 && continue; \
		exit 1; \
	done

ci-job-mujoco-long: assert-docker
	pytest --cov=garage --cov-report=xml --reruns 1 -m 'mujoco_long and not flaky' --durations=20
	for i in {1..5}; do \
		bash <(curl -s https://codecov.io/bash --retry 5) -Z && break \
			|| echo 'Retrying...' && sleep 30 && continue; \
		exit 1; \
	done

ci-job-nightly: assert-docker
	pytest --reruns 1 -m nightly

ci-job-verify-envs: assert-docker ci-job-verify-envs-pipenv ci-job-verify-envs-conda

ci-job-verify-envs-conda: assert-docker
ci-job-verify-envs-conda: CONDA_ROOT := $$HOME/miniconda
ci-job-verify-envs-conda: CONDA := $(CONDA_ROOT)/bin/conda
ci-job-verify-envs-conda: GARAGE_BIN = $(CONDA_ROOT)/envs/garage-ci/bin
ci-job-verify-envs-conda:
	touch $(MJKEY_PATH)
	wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
	bash miniconda.sh -b -p $(CONDA_ROOT)
	hash -r
	$(CONDA) config --set always_yes yes --set changeps1 no
	# Related issue: https://github.com/conda/conda/issues/9105
	# Fix in conda: https://github.com/conda/conda/pull/9014
	# https://repo.continuum.io/miniconda/ doesn't have the script for 4.7.12 yet,
	# so CI fetches 4.7.10 and runs into the above issue when trying to update conda
	$(CONDA) install -c anaconda setuptools
	$(CONDA) update -q conda
	$(CONDA) init
	# Useful for debugging any issues with conda
	$(CONDA) info -a
	$(CONDA) create -n garage-ci python=3.6 pip -y
	$(GARAGE_BIN)/pip install --upgrade pip setuptools
	$(GARAGE_BIN)/pip install dist/garage.tar.gz[all,dev]
	# pylint will verify all imports work
	$(GARAGE_BIN)/pylint --disable=all --enable=import-error garage

# The following two lines remove the Dockerfile's built-in virtualenv from the
# path, so we can test with pipenv directly
ci-job-verify-envs-pipenv: assert-docker
ci-job-verify-envs-pipenv: export PATH=$(shell echo $$PATH_NO_VENV)
ci-job-verify-envs-pipenv: export VIRTUAL_ENV=
ci-job-verify-envs-pipenv: export PIPENV_MAX_RETRIES=2  # number of retries for network requests. Default is 0
ci-job-verify-envs-pipenv:
	touch $(MJKEY_PATH)
	pip3 install --upgrade pip setuptools
	pip3 install pipenv
	pipenv --python=3.6
	pipenv install dist/garage.tar.gz[all,dev]
	pipenv graph
	# pylint will verify all imports work
	pipenv run pylint --disable=all --enable=import-error garage
	@echo "Frozen dependencies:"
	pipenv run pip freeze

ci-deploy-docker: assert-travis
	echo "${DOCKER_API_KEY}" | docker login -u "${DOCKER_USERNAME}" \
		--password-stdin
	docker tag "${TAG}" rlworkgroup/garage-ci:latest
	docker push rlworkgroup/garage-ci

build-ci: TAG ?= rlworkgroup/garage-ci:latest
build-ci: assert-docker-version docker/Dockerfile
	docker build \
		--cache-from rlworkgroup/garage-ci:latest \
		--build-arg BUILDKIT_INLINE_CACHE=1 \
		-f docker/Dockerfile \
		--target garage-dev \
		-t ${TAG} \
		${BUILD_ARGS} .

build-dev: TAG ?= rlworkgroup/garage-dev:latest
build-dev: assert-docker-version docker/Dockerfile
	docker build \
		-f docker/Dockerfile \
		--cache-from rlworkgroup/garage-dev:latest \
		--cache-from rlworkgroup/garage:latest \
		--target garage-dev \
		--build-arg user="$(USER)" \
		--build-arg uid="$(shell id -u)" \
		-t ${TAG} \
		${BUILD_ARGS} .

build-dev-nvidia: TAG ?= rlworkgroup/garage-dev-nvidia:latest
build-dev-nvidia: PARENT_IMAGE ?= nvidia/cuda:11.0-cudnn8-runtime-ubuntu18.04
build-dev-nvidia: assert-docker-version docker/Dockerfile
	docker build \
		-f docker/Dockerfile \
		--cache-from rlworkgroup/garage-dev-nvidia:latest \
		--cache-from rlworkgroup/garage-nvidia:latest \
		--target garage-dev-nvidia \
		-t ${TAG} \
		--build-arg user="$(USER)" \
		--build-arg uid="$(shell id -u)" \
		--build-arg PARENT_IMAGE=${PARENT_IMAGE} \
		${BUILD_ARGS} .

run-ci: ## Run the CI Docker container (only used in TravisCI)
run-ci: TAG ?= rlworkgroup/garage-ci
run-ci:
	docker run \
		-e CODECOV_TOKEN \
		-e TRAVIS_BRANCH \
		-e TRAVIS_PULL_REQUEST \
		-e TRAVIS_COMMIT_RANGE \
		-e TRAVIS \
		-e MJKEY \
		-e GARAGE_GH_TOKEN \
		--memory 7500m \
		--memory-swap 7500m \
		${RUN_ARGS} \
		${TAG} ${RUN_CMD}

run-dev: ## Run the Docker container for headless machines
run-dev: CONTAINER_NAME ?= ''
run-dev: user ?= $$USER
run-dev: ensure-data-path-exists build-dev
	docker run \
		-it \
		--rm \
		-v $(DATA_PATH)/$(CONTAINER_NAME):/home/$(user)/code/garage/data \
		-e MJKEY="$$(cat $(MJKEY_PATH))" \
		--name $(CONTAINER_NAME) \
		${RUN_ARGS} \
		rlworkgroup/garage-dev ${RUN_CMD}

run-dev-nvidia: ## Run the Docker container for machines with NVIDIA GPUs
run-dev-nvidia: ## Requires https://github.com/NVIDIA/nvidia-container-runtime and NVIDIA driver 440+
run-dev-nvidia: CONTAINER_NAME ?= ''
run-dev-nvidia: user ?= $$USER
run-dev-nvidia: GPUS ?= "all"
run-dev-nvidia: ensure-data-path-exists build-dev-nvidia
	xhost +local:docker
	docker run \
		-it \
		--rm \
		--gpus $(GPUS) \
		-v /tmp/.X11-unix:/tmp/.X11-unix \
		-v $(DATA_PATH)/$(CONTAINER_NAME):/home/$(user)/code/garage/data \
		-e DISPLAY=$(DISPLAY) \
		-e QT_X11_NO_MITSHM=1 \
		-e MJKEY="$$(cat $(MJKEY_PATH))" \
		--name $(CONTAINER_NAME) \
		${RUN_ARGS} \
		rlworkgroup/garage-dev-nvidia ${RUN_CMD}

run-dev-nvidia-headless: ## Run the Docker container for machines with NVIDIA GPUs in headless mode
run-dev-nvidia-headless: ## Requires https://github.com/NVIDIA/nvidia-container-runtime and NVIDIA driver 440+
run-dev-nvidia-headless: CONTAINER_NAME ?= ''
run-dev-nvidia-headless: user ?= $$USER
run-dev-nvidia-headless: GPUS ?= "all"
run-dev-nvidia-headless: ensure-data-path-exists build-dev-nvidia
	docker run \
		-it \
		--rm \
		--gpus $(GPUS) \
		-v $(DATA_PATH)/$(CONTAINER_NAME):/home/$(user)/code/garage/data \
		-e MJKEY="$$(cat $(MJKEY_PATH))" \
		--name $(CONTAINER_NAME) \
		${RUN_ARGS} \
		rlworkgroup/garage-dev-nvidia ${RUN_CMD}

# Checks that we are in a docker container
assert-docker:
	@test -f /proc/1/cgroup && /bin/grep -qa docker /proc/1/cgroup \
		|| (echo 'This recipe is only to be run inside Docker.' && exit 1)

# Checks that we are in a docker container
assert-travis:
ifndef TRAVIS
	@echo 'This recipe is only to be run from TravisCI'
	@exit 1
endif

ensure-data-path-exists:
	mkdir -p $(DATA_PATH)/$(CONTAINER_NAME) || { echo "Cannot create directory $(DATA_PATH)/$(CONTAINER_NAME)"; exit 1; }

# Check that the docker version is 19.03 or higher
assert-docker-version:
	@[[ $(shell docker version -f "{{.Server.Version}}" | cut -d'.' -f 1) > 18 ]] \
		|| { echo "You need docker 19.03 or higher to build garage"; exit 1; }

# Help target
# See https://marmelab.com/blog/2016/02/29/auto-documented-makefile.html
help: ## Display this message
help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| sort \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'
