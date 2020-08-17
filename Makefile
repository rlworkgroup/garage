SHELL := /bin/bash

.PHONY: help test check docs ci-job-normal ci-job-large ci-job-nightly \
	ci-job-verify-envs ci-verify-envs-conda ci-verify-envs-pipenv build-ci \
	build-headless build-nvidia run-ci run-headless run-nvidia assert-docker

.DEFAULT_GOAL := help

# Path in host where the experiment data obtained in the container is stored
DATA_PATH ?= $(shell pwd)/data
# Set the environment variable MJKEY with the contents of the file specified by
# MJKEY_PATH.
MJKEY_PATH ?= ${HOME}/.mujoco/mjkey.txt


build-test: TAG ?= rlworkgroup/garage-test
build-test: docker/Dockerfile
	docker build \
		-f docker/Dockerfile \
		--cache-from rlworkgroup/garage-test:latest \
		--cache-from rlworkgroup/garage-headless:latest \
		--target garage-test-18.04 \
		-t ${TAG} \
		${BUILD_ARGS} \
		.

test:  ## Run the garage-test-18.04 docker target
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

ci-job-precommit: assert-docker docs
	scripts/travisci/check_precommit.sh

ci-job-normal: assert-docker
	[ ! -f $(MJKEY_PATH) ] || mv $(MJKEY_PATH) $(MJKEY_PATH).bak
	pytest --cov=garage --cov-report=xml -m \
	    'not nightly and not huge and not flaky and not large and not mujoco and not mujoco_long' --durations=20
	for i in {1..5}; do \
		bash <(curl -s https://codecov.io/bash --retry 5) -Z && break \
			|| echo 'Retrying...' && sleep 30 && continue; \
		exit 1; \
	done

ci-job-large: assert-docker
	[ ! -f $(MJKEY_PATH) ] || mv $(MJKEY_PATH) $(MJKEY_PATH).bak
	pytest --cov=garage --cov-report=xml -m 'large and not flaky' --durations=20
	for i in {1..5}; do \
		bash <(curl -s https://codecov.io/bash --retry 5) -Z && break \
			|| echo 'Retrying...' && sleep 30 && continue; \
		exit 1; \
	done

ci-job-mujoco: assert-docker
	pytest --cov=garage --cov-report=xml -m 'mujoco and not flaky' --durations=20
	for i in {1..5}; do \
		bash <(curl -s https://codecov.io/bash --retry 5) -Z && break \
			|| echo 'Retrying...' && sleep 30 && continue; \
		exit 1; \
	done

ci-job-mujoco-long: assert-docker
	pytest --cov=garage --cov-report=xml -m 'mujoco_long and not flaky' --durations=20
	for i in {1..5}; do \
		bash <(curl -s https://codecov.io/bash --retry 5) -Z && break \
			|| echo 'Retrying...' && sleep 30 && continue; \
		exit 1; \
	done

ci-job-nightly: assert-docker
	pytest -m nightly

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
	$(CONDA) create -n garage-ci python=3.5 pip -y
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
	pip install --upgrade pip setuptools
	pip install pipenv
	pipenv --python=3.5
	pipenv install dist/garage.tar.gz[all,dev]
	pipenv graph
	# pylint will verify all imports work
	pipenv run pylint --disable=all --enable=import-error garage
	@echo "Frozen dependencies:"
	pipenv run pip freeze

ci-deploy-docker: assert-travis
	echo "${DOCKER_API_KEY}" | docker login -u "${DOCKER_USERNAME}" \
		--password-stdin
	docker push rlworkgroup/garage-ci

build-ci: TAG ?= rlworkgroup/garage-ci:latest
build-ci: docker/Dockerfile
	docker build \
		--cache-from ${TAG} \
		-f docker/Dockerfile \
		--target garage-dev-18.04 \
		-t ${TAG} \
		${BUILD_ARGS} .

build-headless: TAG ?= rlworkgroup/garage-headless:latest
build-headless: docker/Dockerfile
	docker build \
		-f docker/Dockerfile \
		--cache-from rlworkgroup/garage-headless:latest \
		--target garage-dev-18.04 \
		--build-arg user="$(USER)" \
		--build-arg uid="$(shell id -u)" \
		-t ${TAG} \
		${BUILD_ARGS} .

build-nvidia: TAG ?= rlworkgroup/garage-nvidia:latest
build-nvidia: PARENT_IMAGE ?= nvidia/cuda:10.2-cudnn7-runtime-ubuntu18.04
build-nvidia: docker/Dockerfile
	docker build \
		-f docker/Dockerfile \
		--cache-from rlworkgroup/garage-nvidia:latest \
		--target garage-nvidia-18.04 \
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

run-headless: ## Run the Docker container for headless machines
run-headless: CONTAINER_NAME ?= ''
run-headless: user ?= $$USER
run-headless: build-headless
	docker run \
		-it \
		--rm \
		-v $(DATA_PATH)/$(CONTAINER_NAME):/home/$(user)/code/garage/data \
		-e MJKEY="$$(cat $(MJKEY_PATH))" \
		--name $(CONTAINER_NAME) \
		${RUN_ARGS} \
		rlworkgroup/garage-headless ${RUN_CMD}

run-nvidia: ## Run the Docker container for machines with NVIDIA GPUs
run-nvidia: ## Requires https://github.com/NVIDIA/nvidia-container-runtime and CUDA 10.2
run-nvidia: CONTAINER_NAME ?= ''
run-nvidia: user ?= $$USER
run-nvidia: build-nvidia
	xhost +local:docker
	docker run \
		-it \
		--rm \
		--runtime=nvidia \
		-v /tmp/.X11-unix:/tmp/.X11-unix \
		-v $(DATA_PATH)/$(CONTAINER_NAME):/home/$(user)/code/garage/data \
		-e DISPLAY=$(DISPLAY) \
		-e QT_X11_NO_MITSHM=1 \
		-e MJKEY="$$(cat $(MJKEY_PATH))" \
		--name $(CONTAINER_NAME) \
		${RUN_ARGS} \
		rlworkgroup/garage-nvidia ${RUN_CMD}

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

# Help target
# See https://marmelab.com/blog/2016/02/29/auto-documented-makefile.html
help: ## Display this message
help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| sort \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'
