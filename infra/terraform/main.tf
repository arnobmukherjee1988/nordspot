terraform {
  required_version = ">= 1.7"

  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 5.0"
    }
  }

  # Uncomment to store Terraform state in GCS (required for team use):
  # backend "gcs" {
  #   bucket = "nordspot-terraform-state"
  #   prefix = "terraform/state"
  # }
}

provider "google" {
  project = var.project_id
  region  = var.region
}

# ---------------------------------------------------------------------------
# Enable GCP APIs
# ---------------------------------------------------------------------------
resource "google_project_service" "apis" {
  for_each = toset([
    "container.googleapis.com",           # GKE
    "artifactregistry.googleapis.com",    # Docker image registry
    "secretmanager.googleapis.com",       # Secret storage
    "storage.googleapis.com",             # GCS data lake
    "iam.googleapis.com",                 # Service accounts
  ])

  service            = each.value
  disable_on_destroy = false
}

# ---------------------------------------------------------------------------
# GKE cluster
# ---------------------------------------------------------------------------
resource "google_container_cluster" "nordspot" {
  name     = "nordspot-${var.environment}"
  location = var.zone

  # Remove the default node pool immediately - we manage our own below.
  remove_default_node_pool = true
  initial_node_count       = 1

  # Enables Workload Identity so pods can authenticate to GCP APIs
  # without storing service account keys inside the cluster.
  workload_identity_config {
    workload_pool = "${var.project_id}.svc.id.goog"
  }

  depends_on = [google_project_service.apis]
}

resource "google_container_node_pool" "primary" {
  name       = "primary"
  cluster    = google_container_cluster.nordspot.name
  location   = var.zone
  node_count = var.gke_node_count

  node_config {
    machine_type = var.gke_machine_type
    disk_size_gb = 50
    disk_type    = "pd-standard"

    # Workload Identity - pods use this SA to call GCP APIs
    workload_metadata_config {
      mode = "GKE_METADATA"
    }

    oauth_scopes = [
      "https://www.googleapis.com/auth/cloud-platform",
    ]
  }

  management {
    auto_repair  = true
    auto_upgrade = true
  }
}

# ---------------------------------------------------------------------------
# Artifact Registry - Docker images
# ---------------------------------------------------------------------------
resource "google_artifact_registry_repository" "nordspot" {
  location      = var.artifact_registry_location
  repository_id = "nordspot"
  description   = "NordSpot Docker images"
  format        = "DOCKER"

  depends_on = [google_project_service.apis]
}

# ---------------------------------------------------------------------------
# GCS bucket - Bronze data lake
# ---------------------------------------------------------------------------
resource "google_storage_bucket" "bronze" {
  name          = var.gcs_bucket_name
  location      = var.region
  force_destroy = false   # prevent accidental deletion in prod

  versioning {
    enabled = true
  }

  lifecycle_rule {
    # Move objects older than 90 days to Nearline (cheaper storage)
    condition {
      age = 90
    }
    action {
      type          = "SetStorageClass"
      storage_class = "NEARLINE"
    }
  }

  uniform_bucket_level_access = true
}

# ---------------------------------------------------------------------------
# Service account for the GKE workloads
# ---------------------------------------------------------------------------
resource "google_service_account" "nordspot_workload" {
  account_id   = "nordspot-workload"
  display_name = "NordSpot GKE Workload SA"
  description  = "Used by GKE pods via Workload Identity to access GCP APIs"
}

resource "google_project_iam_member" "workload_gcs" {
  project = var.project_id
  role    = "roles/storage.objectAdmin"
  member  = "serviceAccount:${google_service_account.nordspot_workload.email}"
}

resource "google_project_iam_member" "workload_secret_accessor" {
  project = var.project_id
  role    = "roles/secretmanager.secretAccessor"
  member  = "serviceAccount:${google_service_account.nordspot_workload.email}"
}

resource "google_project_iam_member" "workload_artifact_reader" {
  project = var.project_id
  role    = "roles/artifactregistry.reader"
  member  = "serviceAccount:${google_service_account.nordspot_workload.email}"
}

# Workload Identity binding - allows the Kubernetes SA to impersonate the GCP SA
resource "google_service_account_iam_member" "workload_identity_binding" {
  service_account_id = google_service_account.nordspot_workload.name
  role               = "roles/iam.workloadIdentityUser"
  member             = "serviceAccount:${var.project_id}.svc.id.goog[nordspot/nordspot-workload]"
}

# ---------------------------------------------------------------------------
# GCP Secret Manager - production secrets
# ---------------------------------------------------------------------------
resource "google_secret_manager_secret" "entsoe_api_key" {
  secret_id = "nordspot-entsoe-api-key"

  replication {
    auto {}
  }

  depends_on = [google_project_service.apis]
}

resource "google_secret_manager_secret_version" "entsoe_api_key" {
  secret      = google_secret_manager_secret.entsoe_api_key.id
  secret_data = var.entsoe_api_key
}

resource "google_secret_manager_secret" "clickhouse_password" {
  secret_id = "nordspot-clickhouse-password"

  replication {
    auto {}
  }

  depends_on = [google_project_service.apis]
}

resource "google_secret_manager_secret_version" "clickhouse_password" {
  secret      = google_secret_manager_secret.clickhouse_password.id
  secret_data = var.clickhouse_password
}

resource "google_secret_manager_secret" "nordspot_api_key" {
  secret_id = "nordspot-api-key"

  replication {
    auto {}
  }

  depends_on = [google_project_service.apis]
}

resource "google_secret_manager_secret_version" "nordspot_api_key" {
  secret      = google_secret_manager_secret.nordspot_api_key.id
  secret_data = var.nordspot_api_key
}
