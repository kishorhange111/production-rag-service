# Common developer commands. Run `make help` to list them.
.DEFAULT_GOAL := help
.PHONY: help install install-dev lint format test check clean serve docker

help:           ## list the available commands
	@grep -E '^[a-zA-Z_-]+:.*## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*## "}; {printf "  %-12s %s\n", $$1, $$2}'

install:        ## this package (lightweight: hashing embedder, extractive answers, no GPU)
	pip install -e .

install-dev:    ## lint/test tooling + this package (editable)
	pip install -e ".[dev]"

lint:           ## ruff lint + format check (what CI runs)
	ruff check .
	ruff format --check .

format:         ## auto-fix lint findings and reformat
	ruff check --fix .
	ruff format .

test:           ## run the test suite (CPU only)
	pytest

check: lint test ## everything CI runs

serve:          ## run the API on http://localhost:8080 over the sample documents
	RAG_DATA_DIR=data/sample uvicorn ragservice.api:app --port 8080

docker:         ## build the image and start it on port 8080
	docker build -t ragservice . && docker run --rm -p 8080:8080 ragservice

clean:          ## remove caches and build artefacts
	rm -rf .pytest_cache .ruff_cache .coverage htmlcov build dist src/*.egg-info
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
