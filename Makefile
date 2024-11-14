help:  ## Show help
	@grep -E '^[.a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'

clean: ## Clean autogenerated files
	rm -rf dist
	find . -type f -name "*.DS_Store" -ls -delete
	find . | grep -E "(__pycache__|\.pyc|\.pyo)" | xargs rm -rf
	find . | grep -E ".pytest_cache" | xargs rm -rf
	find . | grep -E ".ipynb_checkpoints" | xargs rm -rf
	rm -f .coverage

format: ## Run pre-commit hooks
	poetry run pre-commit run -a

sync: ## Merge changes from main branch to your current branch
	git fetch
	git pull

test: ## Run not slow tests
	poetry run pytest -v

test-full: ## Run all tests and coverage.
	poetry run pytest -v --slow

type:
	poetry run mypy .

run: format test-full type

NAME := $(shell whoami)
DOCKER_IMAGE_NAME := $(NAME)/ami:latest

docker-build: ## Build docker image.
	docker build -t $(DOCKER_IMAGE_NAME) --no-cache .

# Docker GPU Option.
USING_GPU_DEVICES := all # Index 0,1,2, ... or device UUID.

GPU_AVAILABLE := $(shell [ -f /proc/driver/nvidia/version ] && echo 1 || echo 0)

ifeq ($(GPU_AVAILABLE),1)
    DOCKER_GPU_OPTION := --gpus device=$(USING_GPU_DEVICES)
else
    DOCKER_GPU_OPTION :=
endif

# Tensorboardなど
DOCKER_PORT_OPTION := --net host

# PulseAudioなど
DOCKER_AUDIO_OPTION := -e PULSE_SERVER=unix:${XDG_RUNTIME_DIR}/pulse/native \
 -v ${XDG_RUNTIME_DIR}/pulse/native:${XDG_RUNTIME_DIR}/pulse/native \
 -v ~/.config/pulse/cookie:/root/.config/pulse/cookie

DOCKER_VOLUME_NAME := ami-$(NAME)

docker-run: ## Run built docker image.
	docker run -itd $(DOCKER_GPU_OPTION) \
	$(DOCKER_PORT_OPTION) \
	--mount type=volume,source=$(DOCKER_VOLUME_NAME),target=/workspace \
	--mount type=bind,source=`pwd`/logs,target=/workspace/logs \
	$(DOCKER_IMAGE_NAME)

docker-run-host: ## Run the built Docker image along with network, camera, and other host OS device access
	docker run -itd $(DOCKER_GPU_OPTION) \
	$(DOCKER_PORT_OPTION) \
	--mount type=volume,source=$(DOCKER_VOLUME_NAME),target=/workspace \
	--mount type=bind,source=`pwd`/logs,target=/workspace/logs \
	--device `v4l2-ctl --list-devices | grep -A 1 'OBS Virtual Camera' | grep -oP '\t\K/dev.*'`:/dev/video0:mwr \
	$(DOCKER_AUDIO_OPTION) \
	$(DOCKER_IMAGE_NAME)

docker-run-unity: ## Run the built Docker image with Unity executables
	docker run -itd $(DOCKER_GPU_OPTION) \
	$(DOCKER_PORT_OPTION) \
	--mount type=volume,source=$(DOCKER_VOLUME_NAME),,target=/workspace \
	--mount type=bind,source=`pwd`/logs,target=/workspace/logs \
	--mount type=bind,source=`pwd`/unity_executables,target=/workspace/unity_executables \
	$(DOCKER_IMAGE_NAME)

DATA_DIR := `pwd`/data
docker-run-with-data:
	docker run -itd $(DOCKER_GPU_OPTION) \
	$(DOCKER_PORT_OPTION) \
	--mount type=volume,source=$(DOCKER_VOLUME_NAME),,target=/workspace \
	--mount type=bind,source=`pwd`/logs,target=/workspace/logs \
	--mount type=bind,source=$(DATA_DIR),target=/workspace/data,readonly \
	$(DOCKER_IMAGE_NAME)

docker-attach: # 一番最後に起動したコンテナにアタッチする。
	@container_id=$$(docker ps --filter "ancestor=$(DOCKER_IMAGE_NAME)" --latest --quiet); \
	if [ -n "$$container_id" ]; then \
		echo "Attaching to container $$container_id"; \
		docker exec -it $$container_id bash; \
	else \
		echo "No running container with image '$(DOCKER_IMAGE_NAME)' found."; \
	fi
