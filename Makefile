.PHONY: up down dev test lint train

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
