SHELL := /bin/bash

.PHONY: help test build-test check docs build-dev \
	build-dev-nvidia run-dev run-dev-nvidia run-dev-nvidia-headless \
	assert-docker ensure-data-path-exists assert-docker-version

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
