.PHONY: help install dev test lint typecheck format run docker-up docker-down clean

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

install: ## Install production dependencies
	pip install -e .

dev: ## Install dev dependencies
	pip install -e ".[dev]"

test: ## Run tests with coverage
	pytest --cov=src --cov-report=term-missing tests/

test-unit: ## Run unit tests only
	pytest tests/unit/ -v

test-integration: ## Run integration tests only
	pytest tests/integration/ -v

lint: ## Run ruff linter
	ruff check src/ tests/

typecheck: ## Run mypy strict type checking
	mypy --strict src/

format: ## Format code with ruff
	ruff format src/ tests/
	ruff check --fix src/ tests/

run: ## Run the API server locally
	uvicorn src.api.main:app --host 0.0.0.0 --port 8080 --reload

docker-up: ## Start all services with docker-compose
	docker-compose up --build -d

docker-down: ## Stop all services
	docker-compose down

docker-logs: ## Tail docker logs
	docker-compose logs -f

clean: ## Remove build artifacts and caches
	rm -rf __pycache__ .pytest_cache .mypy_cache .ruff_cache htmlcov .coverage
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
