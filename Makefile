.PHONY: up down dev test lint train migrate logs api \
        k8s-apply k8s-delete k8s-status k8s-logs \
        tf-init tf-plan tf-apply tf-destroy

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

# -- Kubernetes (K3s locally / GKE in production) ---------------------------

k8s-apply: ## Deploy full stack to the current kubectl context
	kubectl apply -f infra/k8s/

k8s-delete: ## Tear down all nordspot K8s resources (keeps PVCs by default)
	kubectl delete -f infra/k8s/ --ignore-not-found

k8s-status: ## Show pods, services, and ingress in the nordspot namespace
	kubectl get pods,svc,ingress -n nordspot

k8s-logs:  ## Tail logs for a service: make k8s-logs SERVICE=api
	kubectl logs -n nordspot -l app=$(SERVICE) -f --tail=100

# -- Terraform (GCP infrastructure) -----------------------------------------

tf-init:   ## Initialise Terraform (run once after checkout)
	cd infra/terraform && terraform init

tf-plan:   ## Show what Terraform will create/change/destroy
	cd infra/terraform && terraform plan

tf-apply:  ## Apply Terraform plan (creates GCP infrastructure)
	cd infra/terraform && terraform apply

tf-destroy: ## Destroy all GCP infrastructure (irreversible - use with caution)
	cd infra/terraform && terraform destroy
