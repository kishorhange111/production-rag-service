"""Structural validation tests for Docker Compose, Kubernetes, and Helm deployment manifests."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_production_docker_compose_has_observability_and_healthcheck():
    compose = (ROOT / "docker-compose.prod.yml").read_text(encoding="utf-8")
    for token in ("prometheus:", "grafana:", "vllm:", "healthcheck:", "no-new-privileges:true", "/health"):
        assert token in compose


def test_kubernetes_manifests_enforce_security_and_probes():
    k8s_dir = ROOT / "deploy" / "k8s"
    for fname in ("deployment.yaml", "service.yaml", "hpa.yaml", "pdb.yaml", "configmap.yaml"):
        assert (k8s_dir / fname).is_file()

    deploy = (k8s_dir / "deployment.yaml").read_text(encoding="utf-8")
    for token in (
        "runAsNonRoot: true",
        "readOnlyRootFilesystem: true",
        "allowPrivilegeEscalation: false",
        "livenessProbe:",
        "readinessProbe:",
        "prometheus.io/scrape",
    ):
        assert token in deploy

    hpa = (k8s_dir / "hpa.yaml").read_text(encoding="utf-8")
    assert "HorizontalPodAutoscaler" in hpa and "averageUtilization: 70" in hpa


def test_helm_chart_structure_and_templates():
    chart_dir = ROOT / "deploy" / "helm" / "rag-service"
    assert (chart_dir / "Chart.yaml").is_file()
    assert (chart_dir / "values.yaml").is_file()
    for tmpl in ("deployment.yaml", "service.yaml", "hpa.yaml", "configmap.yaml"):
        assert (chart_dir / "templates" / tmpl).is_file()
    chart = (chart_dir / "Chart.yaml").read_text(encoding="utf-8")
    assert "apiVersion: v2" in chart and "name: rag-service" in chart
