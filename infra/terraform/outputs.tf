output "gke_cluster_name" {
  description = "GKE cluster name"
  value       = google_container_cluster.nordspot.name
}

output "gke_cluster_endpoint" {
  description = "GKE cluster API server endpoint"
  value       = google_container_cluster.nordspot.endpoint
  sensitive   = true
}

output "artifact_registry_url" {
  description = "Docker image registry URL"
  value       = "${var.artifact_registry_location}-docker.pkg.dev/${var.project_id}/nordspot"
}

output "gcs_bucket_name" {
  description = "Bronze data lake bucket name"
  value       = google_storage_bucket.bronze.name
}

output "workload_sa_email" {
  description = "GKE workload service account email"
  value       = google_service_account.nordspot_workload.email
}

output "kubectl_config_command" {
  description = "Command to configure kubectl for this cluster"
  value       = "gcloud container clusters get-credentials ${google_container_cluster.nordspot.name} --zone ${var.zone} --project ${var.project_id}"
}
