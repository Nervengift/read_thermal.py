#* Variables
SHELL := /usr/bin/env bash
PYTHON := python
PYTHONPATH := `pwd`

#* Docker variables
IMAGE := sparrow-flir
VERSION := latest

.PHONY: test
test:
	PYTHONPATH=$(PYTHONPATH) poetry run pytest -c pyproject.toml --cov=sparrow_flir sparrow_flir/

#* Formatters/Linters
.PHONY: codestyle
codestyle:
	poetry run pyupgrade --exit-zero-even-if-changed --py39-plus **/*.py
	poetry run isort --settings-path pyproject.toml sparrow_flir
	poetry run black --config pyproject.toml sparrow_flir

.PHONY: check-codestyle
check-codestyle:
	poetry run isort --diff --check-only sparrow_flir
	poetry run black --diff --check sparrow_flir
	poetry run pylint sparrow_flir

.PHONY: mypy
mypy:
	poetry run mypy --config-file pyproject.toml sparrow_flir

.PHONY: check-safety
check-safety:
	poetry check
	poetry run safety check --full-report
	poetry run bandit -ll --recursive sparrow_flir

#* Docker
# Example: make docker-build VERSION=latest
# Example: make docker-build IMAGE=some_name VERSION=0.1.0
.PHONY: docker-build
docker-build:
	@echo Building docker $(IMAGE):$(VERSION) ...
	docker build \
		-t $(IMAGE):$(VERSION) . \
		--no-cache

# Example: make docker-remove VERSION=latest
# Example: make docker-remove IMAGE=some_name VERSION=0.1.0
.PHONY: docker-remove
docker-remove:
	@echo Removing docker $(IMAGE):$(VERSION) ...
	docker rmi -f $(IMAGE):$(VERSION)

.PHONY: branchify
branchify:
ifneq ($(shell git rev-parse --abbrev-ref HEAD),main)
	poetry version $(shell poetry version -s).dev$(shell date +%s)
endif

.PHONY: publish
publish: branchify
	poetry publish --build --repository sparrow