from __future__ import annotations

import json
from typing import Any, Dict
import io
from django.utils import timezone

import os
import openpyxl
import pandas as pd
from django.contrib.auth.decorators import login_required
from django.core.exceptions import PermissionDenied
from django.http import FileResponse, HttpRequest, HttpResponse
from django.shortcuts import get_object_or_404, redirect, render
from django.views.decorators.http import require_http_methods
from rest_framework.exceptions import ValidationError as DRFValidationError
from django.contrib import messages


from apps.datasets.models import Project, Dataset, DatasetVersion
from apps.ml_models.models import Algorithm, Pipeline, PipelineStep, Experiment, MLTaskType, ModelArtifact, EvaluationResult, ManualFeature
from apps.api.v1.serializers import PipelineStepSerializer, ExperimentSerializer, PipelineSerializer
from apps.ml_models.services.experiment_service import run_experiment_threaded
from apps.visualization.services import generate_dataset_plots, plot_project_model_comparison
from apps.visualization.models import PlotArtifact
from apps.ml_models.services.predict_service import predict_csv_for_experiment
from apps.ml_models.services.report_generator import generate_experiment_report_pdf


def _ensure_owner(obj, user):
    owner_id = getattr(obj, "owner_id", None)
    if owner_id is None:
        if hasattr(obj, "dataset") and getattr(obj.dataset, "owner_id", None) != user.id:
            raise PermissionDenied
        return
    if owner_id != user.id:
        raise PermissionDenied


def _plot_url(plot: PlotArtifact) -> str | None:
    if hasattr(plot, "image_file") and getattr(plot, "image_file", None):
        return plot.image_file.url
    # Fallbacks
    if hasattr(plot, "image") and getattr(plot, "image", None):
        return plot.image.url
    if hasattr(plot, "file") and getattr(plot, "file", None):
        return plot.file.url
    return None


@login_required
def dashboard(request: HttpRequest) -> HttpResponse:
    projects = Project.objects.filter(owner=request.user).order_by("-updated_at")[:10]
    recent_datasets = Dataset.objects.filter(owner=request.user).order_by("-updated_at")[:10]
    recent_experiments = Experiment.objects.filter(owner=request.user).order_by("-created_at")[:10]
    return render(request, "dashboard.html", {
        "projects": projects,
        "recent_datasets": recent_datasets,
        "recent_experiments": recent_experiments,
    })


@login_required
def project_detail(request: HttpRequest, project_id: int) -> HttpResponse:
    project = get_object_or_404(Project, id=project_id, owner=request.user)
    datasets = project.datasets.order_by("-updated_at")
    pipelines = project.pipelines.order_by("-updated_at")
    experiments = project.experiments.order_by("-created_at")[:30]
    return render(request, "projects/detail.html", {
        "project": project,
        "datasets": datasets,
        "pipelines": pipelines,
        "experiments": experiments,
    })


@login_required
def project_comparison(request: HttpRequest, project_id: int) -> HttpResponse:
    """
    Renders a model comparison bar chart and a metrics table for all experiments in a project.
    """
    project = get_object_or_404(Project, id=project_id, owner=request.user)

    # Generate/update comparison plot (Accuracy for classification, R2 for regression)
    plot_obj = plot_project_model_comparison(project=project, owner=request.user)

    plot_url = _plot_url(plot_obj) if plot_obj else None

    experiments = (
        Experiment.objects.filter(project=project, owner=request.user)
        .select_related("algorithm", "pipeline")
        .order_by("-created_at")
    )

    rows = []
    for e in experiments:
        ev = EvaluationResult.objects.filter(experiment=e).first()
        metrics = (ev.metrics if ev and isinstance(ev.metrics, dict) else {})
        test = metrics.get("test", {}) if isinstance(metrics, dict) else {}

        primary_metric = None
        if e.pipeline.task_type == MLTaskType.CLASSIFICATION:
            primary_metric = test.get("accuracy")
        elif e.pipeline.task_type == MLTaskType.REGRESSION:
            primary_metric = test.get("r2")

        rows.append({
            "id": e.id,
            "name": e.name,
            "algorithm": e.algorithm.display_name,
            "task": e.pipeline.task_type,
            "status": e.status,
            "accuracy": test.get("accuracy"),
            "f1_macro": test.get("f1_macro"),
            "precision_macro": test.get("precision_macro"),
            "recall_macro": test.get("recall_macro"),
            "rmse": test.get("rmse"),
            "mae": test.get("mae"),
            "r2": test.get("r2"),
            "primary": primary_metric,
        })

    return render(request, "projects/comparison.html", {
        "project": project,
        "plot_url": plot_url,
        "rows": rows,
    })


@login_required
@require_http_methods(["GET", "POST"])
def dataset_upload(request: HttpRequest) -> HttpResponse:
    projects = Project.objects.filter(owner=request.user).order_by("-updated_at")
    
    # Check if we are targeting an existing dataset (New Version Mode)
    target_dataset_id = request.GET.get('dataset_id') or request.POST.get('target_dataset_id')
    target_dataset = None
    if target_dataset_id:
        target_dataset = get_object_or_404(Dataset, id=target_dataset_id, owner=request.user)

    if request.method == "GET":
        return render(request, "datasets/upload.html", {
            "projects": projects, 
            "target_dataset": target_dataset
        })

    # POST Logic
    file = request.FILES.get("file")
    errors = {}
    if not file:
        errors["file"] = "File is required."

    MAX_CSV_MB = 20
    if file and file.size > MAX_CSV_MB * 1024 * 1024:
        errors["file"] = f"File too large (max {MAX_CSV_MB}MB)."

    if errors:
        return render(request, "datasets/upload.html", {
            "projects": projects, 
            "target_dataset": target_dataset,
            "errors": errors
        })

    # Determine Dataset logic
    if target_dataset:
        # We are adding a version to an EXISTING dataset
        dataset = target_dataset
    else:
        # We are creating a NEW dataset
        project_id = request.POST.get("project_id")
        dataset_name = (request.POST.get("dataset_name") or "").strip()
        description = (request.POST.get("description") or "").strip()

        if not project_id: return render(request, "datasets/upload.html", {"projects": projects, "errors": {"project_id": "Required"}})
        if not dataset_name: return render(request, "datasets/upload.html", {"projects": projects, "errors": {"dataset_name": "Required"}})

        project = get_object_or_404(Project, id=int(project_id), owner=request.user)
        dataset, _ = Dataset.objects.get_or_create(
            owner=request.user,
            project=project,
            name=dataset_name,
            defaults={
                "description": description,
                "source_type": "CSV_UPLOAD",
                "external_source": {},
            }
        )
        if description and dataset.description != description:
            dataset.description = description
            dataset.save(update_fields=["description", "updated_at"])

    # --- ROBUST FILE HANDLING LOGIC ---
    ext = os.path.splitext(file.name)[1].lower()
    content = b""
    file_to_save = None

    try:
        if ext in ['.xls', '.xlsx']:
            df = pd.read_excel(file, engine='openpyxl')
            content = df.to_csv(index=False).encode('utf-8')
            filename = os.path.splitext(file.name)[0] + ".csv"
            file_to_save = ContentFile(content, name=filename)
        else:
            content = file.read()
            file.seek(0)
            file_to_save = file
            try:
                pd.read_csv(io.BytesIO(content))
            except Exception:
                file.seek(0)
                df = pd.read_excel(file, engine='openpyxl')
                content = df.to_csv(index=False).encode('utf-8')
                file_to_save = ContentFile(content, name=os.path.splitext(file.name)[0] + ".csv")

    except Exception as e:
        return render(request, "datasets/upload.html", {
            "projects": projects,
            "target_dataset": target_dataset,
            "errors": {"file": f"Error reading file: {str(e)}"}
        })

    sha256 = DatasetVersion.compute_sha256(content)
    latest = DatasetVersion.objects.filter(dataset=dataset).order_by("-version_number").first()
    next_version = (latest.version_number + 1) if latest else 1

    dv = DatasetVersion.objects.create(
        dataset=dataset,
        created_by=request.user,
        version_number=next_version,
        file=file_to_save,
        sha256=sha256,
        row_count=0,
        column_count=0,
        schema_json={},
        profile_json={},
        notes="",
    )

    # Profile
    df = pd.read_csv(dv.file.path)
    dv.row_count = int(df.shape[0])
    dv.column_count = int(df.shape[1])
    dv.schema_json = {"columns": [{"name": c, "dtype": str(df[c].dtype)} for c in df.columns]}
    dv.profile_json = {
        "missing_by_column": {c: int(df[c].isna().sum()) for c in df.columns},
        "unique_by_column": {c: int(df[c].nunique(dropna=True)) for c in df.columns},
    }
    dv.save()

    generate_dataset_plots(dataset_version=dv, df=df, owner=request.user, project=dataset.project)

    return redirect("dataset_version_detail", version_id=dv.id)


@login_required
def dataset_detail(request: HttpRequest, dataset_id: int) -> HttpResponse:
    dataset = get_object_or_404(Dataset, id=dataset_id, owner=request.user)
    versions = dataset.versions.order_by("-version_number")
    return render(request, "datasets/detail.html", {"dataset": dataset, "versions": versions})


@login_required
def dataset_version_detail(request: HttpRequest, version_id: int) -> HttpResponse:
    """
    ENHANCED:
    - Shows first 5 rows preview
    - Highlights data type issues (Check moved to Python side)
    """
    dv = get_object_or_404(DatasetVersion, id=version_id)
    _ensure_owner(dv, request.user)

    columns = (dv.schema_json or {}).get("columns", [])
    profile = dv.profile_json or {}
    missing_by = profile.get("missing_by_column", {}) or {}
    unique_by = profile.get("unique_by_column", {}) or {}

    row_data = []
    for c in columns:
        name = c.get("name")
        dtype = c.get("dtype")
        row_data.append({
            "name": name,
            "dtype": dtype,
            "missing": int(missing_by.get(name, 0)),
            "unique": int(unique_by.get(name, 0)),
        })

    # LOGIC FIX: Calculate if there are issues here in Python
    has_type_issues = any("object" in str(r["dtype"]) for r in row_data)

    # Load first 10 rows for preview
    df = pd.read_csv(dv.file.path, nrows=5)
    preview_rows = df.to_dict('records')

    # Fetch dataset-level plots
    plots = PlotArtifact.objects.filter(dataset_version=dv, experiment__isnull=True)
    dv_plot_map: Dict[str, str] = {}
    for p in plots:
        url = _plot_url(p)
        if url:
            dv_plot_map[p.plot_type] = url

    return render(request, "datasets/version_detail.html", {
        "dv": dv,
        "row_data": row_data,
        "preview_rows": preview_rows,
        "dv_plot_map": dv_plot_map,
        "has_type_issues": has_type_issues, # <-- Passed to template
    })

@login_required
@require_http_methods(["GET", "POST"])
def dataset_version_edit_types(request: HttpRequest, version_id: int) -> HttpResponse:
    """
    Allow users to fix data types for columns.
    Critical for datasets where everything is detected as 'object'.
    """
    dv = get_object_or_404(DatasetVersion, id=version_id)
    _ensure_owner(dv, request.user)

    # Load current data
    df = pd.read_csv(dv.file.path)
    
    # Get current schema
    columns = (dv.schema_json or {}).get("columns", [])
    if not columns:
        columns = [{"name": c, "dtype": str(df[c].dtype)} for c in df.columns]

    if request.method == "GET":
        # Show first 10 rows for preview
        preview_data = df.head(10).to_dict('records')
        
        return render(request, "datasets/edit_types.html", {
            "dv": dv,
            "columns": columns,
            "preview_data": preview_data,
            "column_names": list(df.columns),
        })

    # POST - Apply type changes
    type_changes = {}
    for col in df.columns:
        new_type = request.POST.get(f"type_{col}")
        if new_type:
            type_changes[col] = new_type

    errors = {}
    
    # Apply type conversions with cleaning
    for col, new_type in type_changes.items():
        try:
            if new_type == "int64":
                # Remove currency symbols and commas before converting
                clean_col = df[col].astype(str).str.replace(r'[$,]', '', regex=True)
                s = pd.to_numeric(clean_col, errors='raise')
                df[col] = s.astype('int64')

            elif new_type == "float64":
                # Remove currency symbols and commas
                clean_col = df[col].astype(str).str.replace(r'[$,]', '', regex=True)
                df[col] = pd.to_numeric(clean_col, errors='raise')

            elif new_type == "bool":
                # Robust bool conversion
                true_vals = {'true', '1', 'yes', 'y', 't'}
                df[col] = df[col].astype(str).str.lower().apply(lambda x: x in true_vals)

            elif new_type == "datetime64[ns]":
                # Try dayfirst=True for DD/MM/YYYY formats
                try:
                    df[col] = pd.to_datetime(df[col], dayfirst=True, errors='raise')
                except:
                    # Fallback to standard parser
                    df[col] = pd.to_datetime(df[col], errors='raise')

            elif new_type == "category":
                df[col] = df[col].astype('category')

        except Exception as e:
            errors[col] = f"Cannot convert to {new_type}. Error: {str(e)}"

    if errors:
        preview_data = df.head(10).to_dict('records')
        return render(request, "datasets/edit_types.html", {
            "dv": dv,
            "columns": columns,
            "preview_data": preview_data,
            "column_names": list(df.columns),
            "errors": errors,
        })

    # Save updated dataset as new CSV
    csv_bytes = df.to_csv(index=False).encode('utf-8')
    sha256 = DatasetVersion.compute_sha256(csv_bytes)
    
    # Update the dataset version file
    from django.core.files.base import ContentFile
    dv.file.save(f"corrected_{dv.file.name}", ContentFile(csv_bytes), save=False)
    
    # Update schema and profile
    dv.sha256 = sha256
    dv.row_count = int(df.shape[0])
    dv.column_count = int(df.shape[1])
    dv.schema_json = {"columns": [{"name": c, "dtype": str(df[c].dtype)} for c in df.columns]}
    dv.profile_json = {
        "missing_by_column": {c: int(df[c].isna().sum()) for c in df.columns},
        "unique_by_column": {c: int(df[c].nunique(dropna=True)) for c in df.columns},
    }
    dv.notes = (dv.notes or "") + f"\nData types corrected on {timezone.now().strftime('%Y-%m-%d %H:%M')}"
    dv.save()

    # Regenerate plots with corrected types
    from apps.visualization.services import generate_dataset_plots
    generate_dataset_plots(dataset_version=dv, df=df, owner=request.user, project=dv.dataset.project)

    return redirect("dataset_version_detail", version_id=dv.id)


@login_required
@require_http_methods(["GET", "POST"])
def pipeline_wizard(request: HttpRequest, version_id: int) -> HttpResponse:
    dv = get_object_or_404(DatasetVersion, id=version_id)
    _ensure_owner(dv, request.user)

    dataset = dv.dataset
    project = dataset.project
    _ensure_owner(project, request.user)

    cols = [c["name"] for c in (dv.schema_json or {}).get("columns", [])]
    if not cols:
        df = pd.read_csv(dv.file.path, nrows=5)
        cols = list(df.columns)

    task_types = [
        {"value": MLTaskType.REGRESSION, "label": "Regression"},
        {"value": MLTaskType.CLASSIFICATION, "label": "Classification"},
        {"value": MLTaskType.CLUSTERING, "label": "Clustering"},
    ]

    if request.method == "GET":
        return render(request, "pipelines/wizard.html", {
            "project": project,
            "dataset": dataset,
            "dv": dv,
            "columns": cols,
            "task_types": task_types,
        })

    name = (request.POST.get("pipeline_name") or "").strip()
    task_type = request.POST.get("task_type") or ""
    target_column = (request.POST.get("target_column") or "").strip()
    steps_json = request.POST.get("steps_json") or "[]"
    feature_columns_json = request.POST.get("feature_columns_json") or "[]"
    manual_features_json = request.POST.get("manual_features_json") or "[]"

    errors = {}
    if not name:
        errors["pipeline_name"] = "Pipeline name is required."
    if task_type not in (MLTaskType.REGRESSION, MLTaskType.CLASSIFICATION, MLTaskType.CLUSTERING):
        errors["task_type"] = "Valid task type is required."
    if task_type in (MLTaskType.REGRESSION, MLTaskType.CLASSIFICATION) and (not target_column or target_column not in cols):
        errors["target_column"] = "Target column is required for supervised tasks."

    try:
        steps_payload = json.loads(steps_json)
        if not isinstance(steps_payload, list):
            raise ValueError
    except Exception:
        errors["steps_json"] = "Invalid steps JSON."

    try:
        feature_cols = json.loads(feature_columns_json)
        if not isinstance(feature_cols, list):
            raise ValueError
    except Exception:
        errors["feature_columns_json"] = "Invalid feature columns JSON."
        feature_cols = []

    try:
        manual_defs = json.loads(manual_features_json)
        if not isinstance(manual_defs, list):
            raise ValueError
    except Exception:
        errors["manual_features_json"] = "Invalid manual features JSON."
        manual_defs = []

    # Ensure target column isn't included as a feature for supervised
    if task_type in (MLTaskType.REGRESSION, MLTaskType.CLASSIFICATION) and target_column:
        feature_cols = [c for c in feature_cols if c != target_column]

    if errors:
        return render(request, "pipelines/wizard.html", {
            "project": project,
            "dataset": dataset,
            "dv": dv,
            "columns": cols,
            "task_types": task_types,
            "errors": errors,
        })

    pipeline_data = {
        "project": project.id,
        "dataset_version": dv.id,
        "name": name,
        "task_type": task_type,
        "target_column": target_column if task_type != MLTaskType.CLUSTERING else "",
        "feature_columns": feature_cols,  # now supported by UI
        "random_seed": 42,
    }
    pser = PipelineSerializer(data=pipeline_data)
    pser.is_valid(raise_exception=True)
    pipeline = pser.save(owner=request.user)

    for idx, step in enumerate(steps_payload):
        sdata = {
            "pipeline": pipeline.id,
            "step_type": step.get("step_type"),
            "order": idx,
            "enabled": bool(step.get("enabled", True)),
            "config": step.get("config") or {},
        }
        sser = PipelineStepSerializer(data=sdata)
        sser.is_valid(raise_exception=True)
        sser.save()

    # Save manual features (Safe evaluation happens in service during training)
    for mf in manual_defs:
        n = (mf.get("name") or "").strip()
        expr = (mf.get("expression") or "").strip()
        if n and expr:
            ManualFeature.objects.create(pipeline=pipeline, name=n, expression=expr)

    return redirect("pipeline_detail", pipeline_id=pipeline.id)


@login_required
def pipeline_detail(request: HttpRequest, pipeline_id: int) -> HttpResponse:
    pipeline = get_object_or_404(Pipeline, id=pipeline_id, owner=request.user)
    steps = PipelineStep.objects.filter(pipeline=pipeline).order_by("order")
    mfs = ManualFeature.objects.filter(pipeline=pipeline).order_by("created_at") if hasattr(ManualFeature, "created_at") else ManualFeature.objects.filter(pipeline=pipeline)
    return render(request, "pipelines/detail.html", {"pipeline": pipeline, "steps": steps, "manual_features": mfs})


@login_required
@require_http_methods(["GET", "POST"])
def experiment_create(request: HttpRequest, pipeline_id: int) -> HttpResponse:
    pipeline = get_object_or_404(Pipeline, id=pipeline_id, owner=request.user)
    algorithms = Algorithm.objects.all().order_by("family", "display_name")

    if request.method == "GET":
        algo_payload = []
        for a in algorithms:
            algo_payload.append({
                "id": a.id,
                "key": a.key,
                "display_name": a.display_name,
                "family": a.family,
                "supported_tasks": a.supported_tasks,
                "hyperparameter_schema": a.hyperparameter_schema,
                "default_hyperparameters": a.default_hyperparameters,
            })
        return render(request, "experiments/create.html", {
            "pipeline": pipeline,
            "algorithms": algorithms,
            "algorithms_json": json.dumps(algo_payload),
            "task_type": pipeline.task_type,
        })

    name = (request.POST.get("name") or "").strip()
    algorithm_id = request.POST.get("algorithm_id")
    use_cv = request.POST.get("use_cross_validation") == "on"
    cv_folds = int(request.POST.get("cv_folds") or 5)
    test_size = float(request.POST.get("test_size") or 0.2)
    hyperparams_json = request.POST.get("hyperparameters_json") or "{}"

    errors = {}
    if not name:
        errors["name"] = "Experiment name is required."
    if not algorithm_id:
        errors["algorithm_id"] = "Algorithm is required."

    try:
        hyperparams = json.loads(hyperparams_json)
        if not isinstance(hyperparams, dict):
            raise ValueError
    except Exception:
        errors["hyperparameters_json"] = "Invalid hyperparameters JSON."
        hyperparams = {}

    if errors:
        algo_payload = []
        for a in algorithms:
            algo_payload.append({
                "id": a.id,
                "key": a.key,
                "display_name": a.display_name,
                "family": a.family,
                "supported_tasks": a.supported_tasks,
                "hyperparameter_schema": a.hyperparameter_schema,
                "default_hyperparameters": a.default_hyperparameters,
            })
        return render(request, "experiments/create.html", {
            "pipeline": pipeline,
            "algorithms": algorithms,
            "algorithms_json": json.dumps(algo_payload),
            "task_type": pipeline.task_type,
            "errors": errors,
        })

    algo = get_object_or_404(Algorithm, id=int(algorithm_id))

    exp_data = {
        "project": pipeline.project_id,
        "dataset_version": pipeline.dataset_version_id,
        "pipeline": pipeline.id,
        "algorithm": algo.id,
        "name": name,
        "notes": "",
        "hyperparameters": hyperparams,
        "test_size": test_size,
        "use_cross_validation": use_cv,
        "cv_folds": cv_folds,
        "random_seed": pipeline.random_seed,
    }
    ser = ExperimentSerializer(data=exp_data)
    ser.is_valid(raise_exception=True)
    exp = ser.save(owner=request.user)

    run_experiment_threaded(exp.id)
    return redirect("experiment_results", experiment_id=exp.id)


@login_required
def experiment_results(request: HttpRequest, experiment_id: int) -> HttpResponse:
    exp = get_object_or_404(
        Experiment.objects.select_related("algorithm", "pipeline", "project"),
        id=experiment_id,
        owner=request.user,
    )
    evaluation = EvaluationResult.objects.filter(experiment=exp).first()
    artifact = ModelArtifact.objects.filter(experiment=exp).first()

    metrics = evaluation.metrics if evaluation else {}
    confusion = evaluation.confusion_matrix if evaluation else {}

    test_metrics = metrics.get("test", {}) if isinstance(metrics, dict) else {}
    cv_metrics = metrics.get("cv", {}) if isinstance(metrics, dict) else {}

    cm_matrix = (confusion or {}).get("matrix", [])

    plots = PlotArtifact.objects.filter(experiment=exp).order_by("created_at")
    plot_map: Dict[str, str] = {}
    for p in plots:
        url = _plot_url(p)
        if url:
            plot_map[p.plot_type] = url

    return render(request, "experiments/results.html", {
        "exp": exp,
        "evaluation": evaluation,
        "artifact": artifact,
        "test_metrics": test_metrics,
        "cv_metrics": cv_metrics,
        "cm_matrix": cm_matrix,
        "plot_map": plot_map,
    })


@login_required
def download_model(request: HttpRequest, experiment_id: int) -> HttpResponse:
    exp = get_object_or_404(Experiment, id=experiment_id, owner=request.user)
    artifact = get_object_or_404(ModelArtifact, experiment=exp)
    return FileResponse(artifact.file.open("rb"), as_attachment=True, filename=artifact.file.name.split("/")[-1])


@login_required
@require_http_methods(["GET", "POST"])
def predict_view(request: HttpRequest, experiment_id: int) -> HttpResponse:
    exp = get_object_or_404(Experiment.objects.select_related("project", "algorithm"), id=experiment_id, owner=request.user)

    if request.method == "GET":
        return render(request, "experiments/predict.html", {"exp": exp})

    file = request.FILES.get("file")
    if not file:
        return render(request, "experiments/predict.html", {"exp": exp, "errors": {"file": "CSV file is required."}})

    try:
        csv_bytes = predict_csv_for_experiment(exp, file)
    except DRFValidationError as e:
        return render(request, "experiments/predict.html", {
            "exp": exp,
            "errors": e.detail if isinstance(e.detail, dict) else {"error": str(e)}
        })

    filename = f"predictions_experiment_{exp.id}.csv"
    resp = HttpResponse(csv_bytes, content_type="text/csv; charset=utf-8")
    resp["Content-Disposition"] = f'attachment; filename="{filename}"'
    return resp


@login_required
def download_report_pdf(request: HttpRequest, experiment_id: int) -> HttpResponse:
    exp = get_object_or_404(Experiment.objects.select_related("project", "algorithm"), id=experiment_id, owner=request.user)

    pdf_bytes = generate_experiment_report_pdf(exp)
    filename = f"experiment_report_{exp.id}.pdf"

    resp = HttpResponse(pdf_bytes, content_type="application/pdf")
    resp["Content-Disposition"] = f'attachment; filename="{filename}"'
    return resp



@login_required
def dataset_create_manual(request):
    projects = Project.objects.filter(owner=request.user).order_by("-updated_at")

    if request.method == "GET":
        return render(request, "datasets/create_manual.html", {"projects": projects})

    project_id = request.POST.get("project_id")
    dataset_name = (request.POST.get("dataset_name") or "").strip()
    description = (request.POST.get("description") or "").strip()
    rows_json = request.POST.get("rows_json")
    columns_json = request.POST.get("columns_json")

    errors = {}
    if not project_id:
        errors["project_id"] = "Project is required."
    if not dataset_name:
        errors["dataset_name"] = "Dataset name is required."
    if not rows_json or not columns_json:
        errors["rows_json"] = "Dataset rows and columns are required."

    if errors:
        return render(request, "datasets/create_manual.html", {"projects": projects, "errors": errors})

    # Parse the JSON data
    try:
        columns = json.loads(columns_json)
        rows = json.loads(rows_json)
    except ValueError as e:
        errors["json_parse"] = f"Error parsing JSON data: {str(e)}"
        return render(request, "datasets/create_manual.html", {"projects": projects, "errors": errors})

    # Create the dataset as a DataFrame
    data = {col["name"]: [] for col in columns}
    for row in rows:
        for col in columns:
            data[col["name"]].append(row.get(col["name"], None))
    df = pd.DataFrame(data)

    # Simulate CSV upload for DatasetVersion compatibility
    csv_file = df.to_csv(index=False)
    sha256 = DatasetVersion.compute_sha256(csv_file.encode())
    dataset = Dataset.objects.create(
        owner=request.user,
        project_id=int(project_id),
        name=dataset_name,
        description=description,
    )
    dataset_version = DatasetVersion.objects.create(
        dataset=dataset,
        created_by=request.user,
        version_number=1,
        file=None,  # Simulating CSV upload without file input
        sha256=sha256,
        row_count=df.shape[0],
        column_count=df.shape[1],
        schema_json={"columns": [{"name": c, "dtype": str(df[c].dtype)} for c in df.columns]},
        profile_json={},
        notes="",
    )

    return redirect("dataset_version_detail", version_id=dataset_version.id)

# ==================== NEW VIEWS FOR PHASE 12 ====================

@login_required
@require_http_methods(["GET", "POST"])
def project_create(request: HttpRequest) -> HttpResponse:
    """
    Simple form to create a new Project.
    """
    if request.method == "GET":
        return render(request, "projects/create.html", {})

    name = (request.POST.get("name") or "").strip()
    description = (request.POST.get("description") or "").strip()

    errors = {}
    if not name:
        errors["name"] = "Project name is required."

    if errors:
        return render(request, "projects/create.html", {"errors": errors, "name": name, "description": description})

    project = Project.objects.create(
        owner=request.user,
        name=name,
        description=description,
    )

    return redirect("project_detail", project_id=project.id)


def landing_page(request: HttpRequest) -> HttpResponse:
    """
    Public landing page with Hero section.
    Shows 'Go to Dashboard' if logged in, or 'Login/Signup' if not.
    """
    return render(request, "landing.html", {})


def docs_view(request: HttpRequest) -> HttpResponse:
    """
    Documentation page explaining the workflow.
    """
    return render(request, "docs.html", {})

@login_required
@require_http_methods(["GET", "POST"])
def experiment_create_quick(request: HttpRequest, pipeline_id: int) -> HttpResponse:
    """
    Quick experiment creation - reuses existing pipeline.
    Simplified form: just select algorithm and adjust hyperparameters.
    """
    pipeline = get_object_or_404(Pipeline, id=pipeline_id, owner=request.user)
    algorithms = Algorithm.objects.all().order_by("family", "display_name")

    # Filter algorithms by task type compatibility
    compatible_algorithms = [a for a in algorithms if pipeline.task_type in a.supported_tasks]

    if request.method == "GET":
        algo_payload = []
        for a in compatible_algorithms:
            algo_payload.append({
                "id": a.id,
                "key": a.key,
                "display_name": a.display_name,
                "family": a.family,
                "supported_tasks": a.supported_tasks,
                "hyperparameter_schema": a.hyperparameter_schema,
                "default_hyperparameters": a.default_hyperparameters,
            })
        return render(request, "experiments/create_quick.html", {
            "pipeline": pipeline,
            "algorithms": compatible_algorithms,
            "algorithms_json": json.dumps(algo_payload),
            "task_type": pipeline.task_type,
        })

    # POST - same logic as regular experiment creation
    name = (request.POST.get("name") or "").strip()
    algorithm_id = request.POST.get("algorithm_id")
    use_cv = request.POST.get("use_cross_validation") == "on"
    cv_folds = int(request.POST.get("cv_folds") or 5)
    test_size = float(request.POST.get("test_size") or 0.2)
    hyperparams_json = request.POST.get("hyperparameters_json") or "{}"

    errors = {}
    if not name:
        errors["name"] = "Experiment name is required."
    if not algorithm_id:
        errors["algorithm_id"] = "Algorithm is required."

    try:
        hyperparams = json.loads(hyperparams_json)
        if not isinstance(hyperparams, dict):
            raise ValueError
    except Exception:
        errors["hyperparameters_json"] = "Invalid hyperparameters JSON."
        hyperparams = {}

    if errors:
        algo_payload = []
        for a in compatible_algorithms:
            algo_payload.append({
                "id": a.id,
                "key": a.key,
                "display_name": a.display_name,
                "family": a.family,
                "supported_tasks": a.supported_tasks,
                "hyperparameter_schema": a.hyperparameter_schema,
                "default_hyperparameters": a.default_hyperparameters,
            })
        return render(request, "experiments/create_quick.html", {
            "pipeline": pipeline,
            "algorithms": compatible_algorithms,
            "algorithms_json": json.dumps(algo_payload),
            "task_type": pipeline.task_type,
            "errors": errors,
        })

    algo = get_object_or_404(Algorithm, id=int(algorithm_id))

    exp_data = {
        "project": pipeline.project_id,
        "dataset_version": pipeline.dataset_version_id,
        "pipeline": pipeline.id,
        "algorithm": algo.id,
        "name": name,
        "notes": "",
        "hyperparameters": hyperparams,
        "test_size": test_size,
        "use_cross_validation": use_cv,
        "cv_folds": cv_folds,
        "random_seed": pipeline.random_seed,
    }
    ser = ExperimentSerializer(data=exp_data)
    ser.is_valid(raise_exception=True)
    exp = ser.save(owner=request.user)

    run_experiment_threaded(exp.id)
    return redirect("experiment_results", experiment_id=exp.id)


# ==================== DELETE VIEWS ====================

@login_required
def project_delete(request: HttpRequest, project_id: int) -> HttpResponse:
    project = get_object_or_404(Project, id=project_id, owner=request.user)
    
    if request.method == "POST":
        project.delete()
        messages.success(request, f"Project '{project.name}' and all its resources have been deleted.")
        return redirect("dashboard")
    
    return render(request, "common/confirm_delete.html", {
        "object": project, 
        "type": "Project",
        "warning": "This will permanently delete all datasets, pipelines, experiments, and trained models inside this project."
    })

@login_required
def dataset_delete(request: HttpRequest, dataset_id: int) -> HttpResponse:
    dataset = get_object_or_404(Dataset, id=dataset_id, owner=request.user)
    project_id = dataset.project.id
    
    if request.method == "POST":
        dataset.delete()
        messages.success(request, f"Dataset '{dataset.name}' has been deleted.")
        return redirect("project_detail", project_id=project_id)
    
    return render(request, "common/confirm_delete.html", {
        "object": dataset, 
        "type": "Dataset",
        "warning": "This will delete all versions, pipelines, and experiments associated with this dataset."
    })

@login_required
def experiment_delete(request: HttpRequest, experiment_id: int) -> HttpResponse:
    exp = get_object_or_404(Experiment, id=experiment_id, owner=request.user)
    project_id = exp.project.id
    
    if request.method == "POST":
        exp.delete()
        messages.success(request, f"Experiment '{exp.name}' deleted.")
        return redirect("project_detail", project_id=project_id)
        
    return render(request, "common/confirm_delete.html", {
        "object": exp, 
        "type": "Experiment",
        "warning": "This will delete the trained model artifact and all metrics."
    })