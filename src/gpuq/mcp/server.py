"""Optional MCP adapter (spec section 20).

Deliberately thin: every tool delegates to `GPUQService`, the same object the
CLI uses. No scheduling, snapshotting or state logic is duplicated here, and it
never shells out to `gpuq`.

The SDK is an optional extra, so this module must not be imported at CLI start.
"""

from __future__ import annotations

from typing import Any, Literal

from gpuq import __version__
from gpuq.config import Config, load_config
from gpuq.core import GPUQError, GPUQService, JobNotFound, SubmitRequest

TOOL_NAMES = [
    "gpu_submit",
    "gpu_status",
    "gpu_job",
    "gpu_logs",
    "gpu_cancel",
    "gpu_promote",
    "gpu_info",
]

SUBMIT_FOLLOWUP = (
    "Submitted. Continue independent work; inspect with gpu_job/gpu_logs when needed."
)


class MCPUnavailable(ImportError):
    pass


def _require_sdk() -> Any:
    """Return the server class for whichever SDK generation is installed.

    SDK v2 renamed `FastMCP` to `MCPServer`; the decorator and run APIs are
    otherwise compatible with what this adapter uses. v1 is accepted as a
    fallback so an existing environment keeps working.
    """
    try:
        from mcp.server.mcpserver import MCPServer

        return MCPServer
    except ImportError:
        pass
    try:
        from mcp.server.fastmcp import FastMCP  # SDK v1

        return FastMCP
    except ImportError as exc:
        raise MCPUnavailable(
            "the MCP Python SDK is not installed, or its API is not recognised. "
            "Install it with: uv tool install --force --from . --with 'mcp[cli]' gpuq"
        ) from exc


# --------------------------------------------------------------------------
# Tool implementations - plain functions so they are testable without the SDK
# --------------------------------------------------------------------------


def _service(config: Config | None) -> GPUQService:
    service = GPUQService(config or load_config())
    service.ensure_ready()
    return service


def tool_gpu_submit(
    config: Config | None,
    *,
    command: list[str],
    project: str | None = None,
    priority: str = "normal",
    gpus: int = 1,
    cwd: str | None = None,
    snapshot: bool = True,
    label: str | None = None,
    env: dict[str, str] | None = None,
) -> dict[str, Any]:
    service = _service(config)
    try:
        result = service.submit(
            SubmitRequest(
                command=list(command),
                project=project,
                priority=priority,
                gpus=gpus,
                label=label,
                cwd=cwd,
                snapshot=snapshot,
                env=env or {},
            )
        )
    except GPUQError as exc:
        return {"error": str(exc), "submitted": False}
    finally_payload = {
        "job_id": result.job.id,
        "state": result.job.state,
        "project": result.job.project,
        "priority": result.job.priority,
        "backend_job_id": result.backend_job_id,
        "snapshot_commit": result.job.snapshot_commit,
        "queue_position": result.queue_position,
        "message": SUBMIT_FOLLOWUP,
    }
    service.close()
    return finally_payload


def tool_gpu_status(
    config: Config | None,
    *,
    project: str | None = None,
    state: str | None = None,
    limit: int = 30,
    all_jobs: bool = False,
) -> dict[str, Any]:
    service = _service(config)
    try:
        jobs = service.list_jobs(
            all_jobs=all_jobs, project=project, state=state, limit=limit
        )
        payload = {
            "summary": service.status_summary(),
            "jobs": [j.to_dict() for j in jobs],
        }
    except GPUQError as exc:
        payload = {"error": str(exc)}
    service.close()
    return payload


def tool_gpu_job(config: Config | None, *, job_id: int) -> dict[str, Any]:
    service = _service(config)
    try:
        payload = service.job_detail(job_id)
    except JobNotFound as exc:
        payload = {"error": str(exc)}
    service.close()
    return payload


def tool_gpu_logs(
    config: Config | None, *, job_id: int, tail: int = 200
) -> dict[str, Any]:
    service = _service(config)
    try:
        job = service.get_job(job_id)
    except JobNotFound as exc:
        service.close()
        return {"error": str(exc)}

    path = service.resolve_log_path(job)
    if path is None or not path.exists():
        service.close()
        return {
            "job_id": job_id,
            "state": job.state,
            "log": "",
            "message": f"job #{job_id} is {job.state}; no output file yet",
        }
    from collections import deque

    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        lines = deque(fh, maxlen=max(1, tail))
    service.close()
    return {
        "job_id": job_id,
        "state": job.state,
        "log_path": str(path),
        "log": "".join(lines),
    }


def tool_gpu_cancel(
    config: Config | None, *, job_id: int, force: bool = False
) -> dict[str, Any]:
    service = _service(config)
    try:
        payload = service.cancel(job_id, force=force)
    except (JobNotFound, GPUQError) as exc:
        payload = {"error": str(exc)}
    service.close()
    return payload


def tool_gpu_promote(config: Config | None, *, job_id: int) -> dict[str, Any]:
    service = _service(config)
    try:
        payload = service.promote(job_id)
    except (JobNotFound, GPUQError) as exc:
        payload = {"error": str(exc)}
    service.close()
    return payload


def tool_gpu_info(config: Config | None) -> dict[str, Any]:
    service = _service(config)
    payload = {
        "gpuq_version": __version__,
        "gpu": service.gpu_info().to_dict(),
        "summary": service.status_summary(),
    }
    service.close()
    return payload


# --------------------------------------------------------------------------
# Server construction
# --------------------------------------------------------------------------


def build_server(config: Config | None = None) -> Any:
    """Build a FastMCP server exposing the GPUQ tools."""
    server_class = _require_sdk()
    config = config or load_config()

    server = server_class(
        name="gpuq",
        instructions=(
            "Broker for GPU-heavy workloads on this machine. Submit expensive work with "
            "gpu_submit instead of running it directly, then continue with other work; "
            "only one heavy job runs at a time so concurrent agents cannot OOM the GPU."
        ),
    )

    @server.tool(description="Submit a GPU-heavy command to the shared queue and return at once.")
    def gpu_submit(
        command: list[str],
        project: str | None = None,
        priority: Literal["critical", "high", "normal", "low"] = "normal",
        gpus: int = 1,
        cwd: str | None = None,
        snapshot: bool = True,
        label: str | None = None,
        env: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        return tool_gpu_submit(
            config,
            command=command,
            project=project,
            priority=priority,
            gpus=gpus,
            cwd=cwd,
            snapshot=snapshot,
            label=label,
            env=env,
        )

    @server.tool(description="List running, queued and recently finished GPUQ jobs.")
    def gpu_status(
        project: str | None = None,
        state: str | None = None,
        limit: int = 30,
        all_jobs: bool = False,
    ) -> dict[str, Any]:
        return tool_gpu_status(
            config, project=project, state=state, limit=limit, all_jobs=all_jobs
        )

    @server.tool(description="Full detail for one job, including source provenance.")
    def gpu_job(job_id: int) -> dict[str, Any]:
        return tool_gpu_job(config, job_id=job_id)

    @server.tool(description="Tail a job's output.")
    def gpu_logs(job_id: int, tail: int = 200) -> dict[str, Any]:
        return tool_gpu_logs(config, job_id=job_id, tail=tail)

    @server.tool(description="Cancel a queued or running job.")
    def gpu_cancel(job_id: int, force: bool = False) -> dict[str, Any]:
        return tool_gpu_cancel(config, job_id=job_id, force=force)

    @server.tool(description="Move a queued job to the front of the queue.")
    def gpu_promote(job_id: int) -> dict[str, Any]:
        return tool_gpu_promote(config, job_id=job_id)

    @server.tool(description="GPU inventory and queue configuration.")
    def gpu_info() -> dict[str, Any]:
        return tool_gpu_info(config)

    return server


def self_test(config: Config | None = None) -> dict[str, Any]:
    """Build the server in-process and confirm every tool registered."""
    try:
        server = build_server(config)
    except MCPUnavailable as exc:
        return {"ok": False, "error": str(exc), "tools": []}
    except Exception as exc:  # pragma: no cover
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}", "tools": []}

    # `list_tools` is a coroutine function in some SDK builds and plain in
    # others. Decide before calling, so we never create a coroutine we then
    # abandon un-awaited.
    try:
        import inspect

        if inspect.iscoroutinefunction(server.list_tools):
            import anyio

            tools = anyio.run(server.list_tools)
        else:
            tools = server.list_tools()
        names = sorted(tool.name for tool in tools)
    except Exception as exc:  # pragma: no cover - SDK shape differences
        return {"ok": False, "error": f"could not list tools: {exc}", "tools": []}

    missing = sorted(set(TOOL_NAMES) - set(names))
    if missing:
        return {"ok": False, "error": f"missing tools: {missing}", "tools": names}
    return {"ok": True, "error": None, "tools": names}


def serve(config: Config | None = None) -> None:
    """Run the stdio server. stdio only - no network listener (spec section 30)."""
    build_server(config).run(transport="stdio")
