SHELL := /bin/bash

.PHONY: help test check docs build-headless build-nvidia run-headless \
 		run-nvidia assert-docker

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
build-nvidia: PARENT_IMAGE ?= nvidia/cuda:10.1-cudnn7-runtime-ubuntu18.04
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
run-nvidia: ## Requires https://github.com/NVIDIA/nvidia-container-runtime and CUDA 10.1
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

# Help target
# See https://marmelab.com/blog/2016/02/29/auto-documented-makefile.html
help: ## Display this message
help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| sort \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'
