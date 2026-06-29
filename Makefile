.PHONY: up down dev test lint train migrate logs api

up:        ## Start the full local stack
	docker compose up -d

down:      ## Stop all containers
	docker compose down

dev:       ## Start lightweight dev stack (ClickHouse + Redis only)
	docker compose -f docker-compose.dev.yml up -d

test:      ## Run all pytest tests
	pytest tests/ -v

lint:      ## Lint and format check (does not auto-fix)
	ruff check .
	ruff format --check .

train:     ## Run model training
	python -m ml.train

migrate:   ## Create ClickHouse schema (idempotent - safe to re-run)
	PYTHONPATH=. python -m db.schema

logs:      ## Tail logs from all running containers
	docker compose logs -f

api:       ## Run the FastAPI server locally without Docker (dev shortcut)
	uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
