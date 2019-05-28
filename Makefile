SHELL := /bin/bash

.PHONY: help test check docs ci-job-normal ci-job-large ci-job-nightly \
 	build-headless build-nvidia run-ci run-headless run-nvidia

.DEFAULT_GOAL := help

# Path in host where the experiment data obtained in the container is stored
DATA_PATH ?= $(shell pwd)/data
# Set the environment variable MJKEY with the contents of the file specified by
# MJKEY_PATH.
MJKEY_PATH ?= ~/.mujoco/mjkey.txt

test:  ## Run the CI test suite
test: RUN_CMD = nose2 -c setup.cfg -v --with-id -E 'not huge and not flaky'
test: run-headless
	@echo "Running test suite..."

docs:  ## Build HTML documentation
docs:
	@pushd docs && make html && popd

ci-precommit-check:
	scripts/travisci/check_precommit.sh

ci-job-normal: ci-precommit-check docs
	coverage run -m nose2 -c setup.cfg -v --with-id -E \
	    'not nightly and not huge and not flaky and not large'
	coverage xml
	bash <(curl -s https://codecov.io/bash)

ci-job-large:
	coverage run -m nose2 -c setup.cfg -v --with-id -A large
	coverage xml
	bash <(curl -s https://codecov.io/bash)

ci-job-nightly:
	nose2 -c setup.cfg -A nightly

ci-deploy-docker:
	echo "${DOCKER_API_KEY}" | docker login -u "${DOCKER_USERNAME}" \
		--password-stdin
	docker tag "${TAG}" rlworkgroup/garage-ci:latest
	docker push rlworkgroup/garage-ci

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

build-intel: TAG ?= rlworkgroup/garage-intel:latest
build-intel: docker/docker-compose-intel.yml
	TAG=${TAG} \
	docker-compose \
		-f docker/docker-compose-intel.yml \
		build \
		${ADD_ARGS}

run-ci: ## Run the CI Docker container (only used in TravisCI)
run-ci: TAG ?= rlworkgroup/garage-headless:latest
run-ci: build-headless
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

run-intel: ## Run the Docker container for machines with Intel CPUs
run-intel: CONTAINER_NAME ?= garage-intel
run-intel: build-intel
	xhost +local:docker
	docker run \
		-it \
		--rm \
		-v /tmp/.X11-unix:/tmp/.X11-unix \
		-v $(DATA_PATH)/$(CONTAINER_NAME):/root/code/garage/data \
		-e DISPLAY=$(DISPLAY) \
		-e QT_X11_NO_MITSHM=1 \
		-e MJKEY="$$(cat $(MJKEY_PATH))" \
		--name $(CONTAINER_NAME) \
		${ADD_ARGS} \
		rlworkgroup/garage-intel $(RUN_CMD)


# Help target
# See https://marmelab.com/blog/2016/02/29/auto-documented-makefile.html
help: ## Display this message
help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| sort \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'
