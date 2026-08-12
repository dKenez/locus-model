#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_NAME = locus-model
PYTHON_VERSION = 3.12
PYTHON_INTERPRETER = uv run python

#################################################################################
# COMMANDS                                                                      #
#################################################################################

## Set up python interpreter environment
.PHONY: create_environment
create_environment:
	uv python install $(PYTHON_VERSION)

## Install Python Dependencies
.PHONY: requirements
requirements:
	uv sync

## Install Developer Python Dependencies
.PHONY: dev_requirements
dev_requirements:
	uv sync --all-groups --all-extras
	uv run mypy --install-types --non-interactive

## Delete all compiled Python files
.PHONY: clean
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete

#################################################################################
# PROJECT RULES                                                                 #
#################################################################################

## Make Dataset
ifeq ($(delete_existing), 1)
    CFLAGS += --delete-existing
endif
.PHONY: data
data:
	$(PYTHON_INTERPRETER) locus/data/make_dataset.py $(CFLAGS)

#################################################################################
# Documentation RULES                                                           #
#################################################################################

.PHONY: build_documentation
build_documentation: dev_requirements
	uv run mkdocs build --config-file docs/mkdocs.yaml --site-dir build

.PHONY: serve_documentation
serve_documentation: dev_requirements
	uv run mkdocs serve --config-file docs/mkdocs.yaml

#################################################################################
# Self Documenting Commands                                                     #
#################################################################################

.DEFAULT_GOAL := help

define PRINT_HELP_PYSCRIPT
import re, sys

for line in sys.stdin:
	match = re.match(r'^([a-zA-Z_-]+):.*?## (.*)$$', line)
	if match:
		target, help = match.groups()
		print("%-20s %s" % (target, help))
endef
export PRINT_HELP_PYSCRIPT

help:
	@python -c "$$PRINT_HELP_PYSCRIPT" < $(MAKEFILE_LIST)
