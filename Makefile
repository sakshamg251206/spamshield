# Common development tasks. Run `make help` to list them.
.DEFAULT_GOAL := help
.PHONY: help install run train lint format typecheck test check requirements docker clean

# Load .env when present so `make run` honours local configuration.
ifneq (,$(wildcard .env))
include .env
export
endif

help: ## Show this help
	@grep -E '^[a-z-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-10s\033[0m %s\n", $$1, $$2}'

install: ## Install the app and dev tools into .venv
	uv sync
	uv run pre-commit install

run: ## Start the web app on http://localhost:8501
	uv run streamlit run app.py

train: ## Retrain, compare and save the model
	uv run spamshield train

lint: ## Check style and common bugs
	uv run ruff check .
	uv run ruff format --check .

format: ## Auto-format and fix lint issues
	uv run ruff format .
	uv run ruff check --fix .

typecheck: ## Run the static type checker
	uv run mypy

test: ## Run the test suite with coverage
	uv run pytest --cov

check: lint typecheck test ## Everything CI runs

requirements: ## Regenerate requirements.txt from uv.lock
	uv export --frozen --no-dev --no-hashes --quiet -o requirements.txt

docker: ## Build and run the Docker image
	docker build -t spamshield .
	docker run --rm -p 8501:8501 spamshield

clean: ## Remove caches and build output
	rm -rf .pytest_cache .mypy_cache .ruff_cache .coverage htmlcov build dist
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
