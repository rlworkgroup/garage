.PHONY: help test build-ci build-headless build-nvidia run-ci run-headless \
	run-nvidia

.DEFAULT_GOAL := help

# Path in host where the experiment data obtained in the container is stored
DATA_PATH ?= $(shell pwd)/data
# Set the environment variable MJKEY with the contents of the file specified by
# MJKEY_PATH.
MJKEY_PATH ?= ~/.mujoco/mjkey.txt
# Prevent garage from exiting on config by making sure the personal config file
# is already created
CONFIG_PERSONAL := garage/config_personal.py

test:  ## Run the CI test suite
test: RUN_CMD = nose2 -c setup.cfg -E 'not cron_job and not huge and not flaky'
test: run-headless
	@echo "Running test suite..."

build-ci: TAG ?= rlworkgroup/garage-ci:latest
build-ci: docker/docker-compose-ci.yml copy_config_personal
	TAG=${TAG} \
	docker-compose \
		-f docker/docker-compose-ci.yml \
		build \
		${ADD_ARGS}

build-headless: TAG ?= rlworkgroup/garage-headless:latest
build-headless: docker/docker-compose-headless.yml copy_config_personal
	TAG=${TAG} \
	docker-compose \
		-f docker/docker-compose-headless.yml \
		build \
		${ADD_ARGS}

build-nvidia: TAG ?= rlworkgroup/garage-nvidia:latest
build-nvidia: docker/docker-compose-nvidia.yml copy_config_personal
	TAG=${TAG} \
	docker-compose \
		-f docker/docker-compose-nvidia.yml \
		build \
		${ADD_ARGS}

run-ci: ## Run the CI Docker container (only used in TravisCI)
run-ci: TAG ?= rlworkgroup/garage-ci
run-ci:
	docker run \
		-e COVERALLS_REPO_TOKEN \
		-e COVERALLS_SERVICE_NAME \
		-e CODACY_PROJECT_TOKEN \
		-e CC_TEST_REPORTER_ID \
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

copy_config_personal:
ifeq (0, $(shell [ ! -f $(CONFIG_PERSONAL) ]; echo $$? ))
	$(warning No file for personal configuration was found. Copying from \
		garage/config_personal_template.py)
	cp garage/config_personal_template.py garage/config_personal.py
endif


# Help target
# See https://marmelab.com/blog/2016/02/29/auto-documented-makefile.html
help: ## Display this message
help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| sort \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'
