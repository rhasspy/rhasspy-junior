.PHONY: docker

SHELL := bash

# linux/amd64,linux/arm64
DOCKER_PLATFORM ?= linux/amd64
DOCKER_OUTPUT ?=
DOCKER_TAG ?= rhasspy/junior-hass

docker:
	docker buildx build . -f Dockerfile --platform "$(DOCKER_PLATFORM)" --tag "$(DOCKER_TAG)" $(DOCKER_OUTPUT)
