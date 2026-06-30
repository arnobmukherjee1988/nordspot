variable "project_id" {
  description = "GCP project ID"
  type        = string
}

variable "region" {
  description = "GCP region for all resources"
  type        = string
  default     = "europe-north1"   # Finland - lowest latency to Nordic power markets
}

variable "zone" {
  description = "GCP zone for GKE node pool"
  type        = string
  default     = "europe-north1-a"
}

variable "environment" {
  description = "Deployment environment (dev / staging / prod)"
  type        = string
  default     = "prod"

  validation {
    condition     = contains(["dev", "staging", "prod"], var.environment)
    error_message = "environment must be one of: dev, staging, prod"
  }
}

variable "gke_node_count" {
  description = "Number of nodes in the GKE node pool"
  type        = number
  default     = 2
}

variable "gke_machine_type" {
  description = "GCE machine type for GKE nodes"
  type        = string
  default     = "n1-standard-2"   # 2 vCPU, 7.5 GB RAM
}

variable "gcs_bucket_name" {
  description = "Name of the GCS bucket used as the Bronze data lake"
  type        = string
  default     = "nordspot-bronze"
}

variable "artifact_registry_location" {
  description = "Location for the Artifact Registry repository"
  type        = string
  default     = "europe-north1"
}

variable "entsoe_api_key" {
  description = "ENTSO-E Transparency Platform API key"
  type        = string
  sensitive   = true
}

variable "clickhouse_password" {
  description = "ClickHouse admin password"
  type        = string
  sensitive   = true
  default     = "nordspot"
}

variable "nordspot_api_key" {
  description = "API key for NordSpot REST API authentication"
  type        = string
  sensitive   = true
}
