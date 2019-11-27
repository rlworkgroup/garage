SHELL := /bin/bash

.PHONY: help test check docs ci-job-normal ci-job-large ci-job-nightly \
	ci-job-verify-envs ci-verify-conda ci-verify-pipenv build-ci \
	build-headless build-nvidia run-ci run-headless run-nvidia

.DEFAULT_GOAL := help

# Path in host where the experiment data obtained in the container is stored
DATA_PATH ?= $(shell pwd)/data
# Set the environment variable MJKEY with the contents of the file specified by
# MJKEY_PATH.
MJKEY_PATH ?= ~/.mujoco/mjkey.txt

test:  ## Run the CI test suite
test: RUN_CMD = pytest -n $$(($$(nproc)/4)) -v -m 'not huge and not flaky' --durations=0
test: run-headless
	@echo "Running test suite..."

docs:  ## Build HTML documentation
docs:
	@pushd docs && make html && popd

ci-job-precommit:
	scripts/travisci/check_precommit.sh

ci-job-normal: docs
	pytest -n $$(nproc) --cov=garage -v -m \
	    'not nightly and not huge and not flaky and not large' --durations=0
	coverage xml
	bash <(curl -s https://codecov.io/bash)

ci-job-large:
	pytest -n $$(nproc) --cov=garage -v -m large
	coverage xml
	bash <(curl -s https://codecov.io/bash)

ci-job-nightly:
	pytest -n $$(nproc) -v -m nightly

ci-job-verify-envs: ci-verify-conda ci-verify-pipenv

ci-verify-conda: CONDA_ROOT := $$HOME/miniconda
ci-verify-conda: CONDA := $(CONDA_ROOT)/bin/conda
ci-verify-conda: GARAGE_BIN = $(CONDA_ROOT)/envs/garage-ci/bin
ci-verify-conda:
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
	$(GARAGE_BIN)/pip install --upgrade pip
	$(GARAGE_BIN)/pip install dist/garage.tar.gz[all,dev]
	# pylint will verify all imports work
	$(GARAGE_BIN)/pylint --disable=all --enable=import-error garage

# The following two lines remove the Dockerfile's built-in virtualenv from the
# path, so we can test with pipenv directly
ci-verify-pipenv: export PATH=$(shell echo $$PATH | awk -v RS=: -v ORS=: '/venv/ {next} {print}')
ci-verify-pipenv: export VIRTUAL_ENV=
ci-verify-pipenv:
	pip3 install --upgrade pipenv setuptools
	pipenv --three
	pipenv install dist/garage.tar.gz[all,dev]
	pipenv graph
	# pylint will verify all imports work
	pipenv run pylint --disable=all --enable=import-error garage

ci-deploy-docker:
	echo "${DOCKER_API_KEY}" | docker login -u "${DOCKER_USERNAME}" \
		--password-stdin
	docker tag "${TAG}" rlworkgroup/garage-ci:latest
	docker push rlworkgroup/garage-ci

build-ci: TAG ?= rlworkgroup/garage-ci:latest
build-ci: docker/docker-compose-ci.yml
	TAG=${TAG} \
	docker-compose \
		-f docker/docker-compose-ci.yml \
		build \
		${ADD_ARGS}

build-headless: TAG ?= rlworkgroup/garage-headless:latest
build-headless: docker/docker-compose-headless.yml
	TAG=${TAG} \
	docker-compose \
		-f docker/docker-compose-headless.yml \
		build \
		${ADD_ARGS}

build-nvidia: TAG ?= rlworkgroup/garage-nvidia:latest
build-nvidia: docker/docker-compose-nvidia.yml
	TAG=${TAG} \
	docker-compose \
		-f docker/docker-compose-nvidia.yml \
		build \
		${ADD_ARGS}

run-ci: ## Run the CI Docker container (only used in TravisCI)
run-ci: TAG ?= rlworkgroup/garage-ci
run-ci:
	docker run \
		-e TRAVIS_BRANCH \
		-e TRAVIS_PULL_REQUEST \
		-e TRAVIS_COMMIT_RANGE \
		-e TRAVIS \
		-e MJKEY \
		${ADD_ARGS} \
		${TAG} ${RUN_CMD}

run-headless: ## Run the Docker container for headless machines
run-headless: CONTAINER_NAME ?= garage-headless
run-headless: build-headless
	docker run \
		-it \
		--rm \
		-v $(DATA_PATH)/$(CONTAINER_NAME):/root/code/garage/data \
		-e MJKEY="$$(cat $(MJKEY_PATH))" \
		--name $(CONTAINER_NAME) \
		${ADD_ARGS} \
		rlworkgroup/garage-headless $(RUN_CMD)

run-nvidia: ## Run the Docker container for machines with NVIDIA GPUs
run-nvidia: CONTAINER_NAME ?= garage-nvidia
run-nvidia: build-nvidia
	xhost +local:docker
	docker run \
		-it \
		--rm \
		--runtime=nvidia \
		-v /tmp/.X11-unix:/tmp/.X11-unix \
		-v $(DATA_PATH)/$(CONTAINER_NAME):/root/code/garage/data \
		-e DISPLAY=$(DISPLAY) \
		-e QT_X11_NO_MITSHM=1 \
		-e MJKEY="$$(cat $(MJKEY_PATH))" \
		--name $(CONTAINER_NAME) \
		${ADD_ARGS} \
		rlworkgroup/garage-nvidia $(RUN_CMD)

# Help target
# See https://marmelab.com/blog/2016/02/29/auto-documented-makefile.html
help: ## Display this message
help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| sort \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'
