from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict

import torch


def reset_peak_memory_stats_for_device(device: torch.device | None) -> None:
    if device is None or device.type != "cuda" or not torch.cuda.is_available():
        return
    try:
        torch.cuda.reset_peak_memory_stats(device)
    except Exception:
        # Best-effort instrumentation should never break training.
        return


def collect_local_resource_summary(
    *,
    stage: str,
    run_name: str | None,
    config_path: str | None,
    output_dir: Path,
    device: torch.device | None,
    rank: int,
    world_size: int,
    status: str,
    optimizer_step: int,
    global_step: int,
    training_start_time: float | None,
    interrupted: bool = False,
    error_type: str | None = None,
    error_message: str | None = None,
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "stage": stage,
        "run_name": run_name,
        "config_path": config_path,
        "output_dir": str(output_dir),
        "rank": rank,
        "world_size": world_size,
        "device": str(device) if device is not None else None,
        "status": status,
        "interrupted": interrupted,
        "optimizer_step": int(optimizer_step),
        "global_step": int(global_step),
        "recorded_at_unix": time.time(),
    }
    if training_start_time is not None:
        payload["training_wall_time_seconds"] = time.perf_counter() - training_start_time
    if error_type is not None:
        payload["error_type"] = error_type
    if error_message is not None:
        payload["error_message"] = error_message

    if device is not None and device.type == "cuda" and torch.cuda.is_available():
        try:
            torch.cuda.synchronize(device)
        except Exception:
            pass
        try:
            device_props = torch.cuda.get_device_properties(device)
            payload["cuda_device_name"] = device_props.name
            payload["cuda_total_memory_gb"] = device_props.total_memory / (1024 ** 3)
        except Exception:
            payload["cuda_device_name"] = None
            payload["cuda_total_memory_gb"] = None
        try:
            payload["cuda_peak_memory_allocated_gb"] = (
                torch.cuda.max_memory_allocated(device) / (1024 ** 3)
            )
            payload["cuda_peak_memory_reserved_gb"] = (
                torch.cuda.max_memory_reserved(device) / (1024 ** 3)
            )
            payload["cuda_memory_allocated_gb"] = torch.cuda.memory_allocated(device) / (1024 ** 3)
            payload["cuda_memory_reserved_gb"] = torch.cuda.memory_reserved(device) / (1024 ** 3)
        except Exception:
            payload["cuda_peak_memory_allocated_gb"] = None
            payload["cuda_peak_memory_reserved_gb"] = None
            payload["cuda_memory_allocated_gb"] = None
            payload["cuda_memory_reserved_gb"] = None
    else:
        payload["cuda_device_name"] = None
        payload["cuda_total_memory_gb"] = None
        payload["cuda_peak_memory_allocated_gb"] = None
        payload["cuda_peak_memory_reserved_gb"] = None
        payload["cuda_memory_allocated_gb"] = None
        payload["cuda_memory_reserved_gb"] = None

    return payload


def write_local_resource_summary(output_dir: Path, payload: Dict[str, Any], rank: int) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / f"resource_summary_rank{rank}.json"
    summary_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary_path


def try_write_aggregate_resource_summary(output_dir: Path, world_size: int) -> Path | None:
    rank_paths = [output_dir / f"resource_summary_rank{rank}.json" for rank in range(world_size)]
    if not rank_paths or not all(path.exists() for path in rank_paths):
        return None

    summaries = [json.loads(path.read_text(encoding="utf-8")) for path in rank_paths]
    aggregate: Dict[str, Any] = {
        "stage": summaries[0].get("stage"),
        "run_name": summaries[0].get("run_name"),
        "config_path": summaries[0].get("config_path"),
        "output_dir": summaries[0].get("output_dir"),
        "world_size": world_size,
        "num_rank_summaries": len(summaries),
        "optimizer_step": max(int(item.get("optimizer_step", 0)) for item in summaries),
        "global_step": max(int(item.get("global_step", 0)) for item in summaries),
        "status": "completed",
        "interrupted": any(bool(item.get("interrupted", False)) for item in summaries),
        "per_rank_files": [path.name for path in rank_paths],
        "per_rank": summaries,
    }

    if any(item.get("status") == "failed" for item in summaries):
        aggregate["status"] = "failed"
    elif aggregate["interrupted"]:
        aggregate["status"] = "interrupted"

    wall_times = [
        float(item["training_wall_time_seconds"])
        for item in summaries
        if item.get("training_wall_time_seconds") is not None
    ]
    if wall_times:
        aggregate["max_training_wall_time_seconds"] = max(wall_times)

    peak_allocated = [
        float(item["cuda_peak_memory_allocated_gb"])
        for item in summaries
        if item.get("cuda_peak_memory_allocated_gb") is not None
    ]
    peak_reserved = [
        float(item["cuda_peak_memory_reserved_gb"])
        for item in summaries
        if item.get("cuda_peak_memory_reserved_gb") is not None
    ]
    if peak_allocated:
        aggregate["max_cuda_peak_memory_allocated_gb"] = max(peak_allocated)
    if peak_reserved:
        aggregate["max_cuda_peak_memory_reserved_gb"] = max(peak_reserved)

    summary_path = output_dir / "resource_summary.json"
    summary_path.write_text(json.dumps(aggregate, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary_path
