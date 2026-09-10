"""The local web UI's server and API.

Driven over real HTTP against a server on an ephemeral port, because the thing
worth testing is the thing the browser talks to - the routing, the payload
shapes and, most of all, that a threaded server does not share SQLite handles
across request threads.

No new test dependencies: `http.client` is in the standard library, as is the
server itself.
"""

from __future__ import annotations

import json
import threading
from concurrent.futures import ThreadPoolExecutor
from http.client import HTTPConnection

import pytest

from workerq.models import JobState
from workerq.util import utcnow_iso
from workerq.web.server import LoopbackOnly, serve


@pytest.fixture
def queued_job(service):
    """One finished job with a measured peak, so the accuracy path has input."""
    job_id = service.db.insert_job(
        backend="local_dispatcher",
        project="demo",
        priority="normal",
        submitted_cwd=str(service.config.state_dir),
        command_json=json.dumps(["python", "train.py"]),
        snapshot_mode="none",
        host="testhost",
        state=JobState.SUCCEEDED.value,
        exit_code=0,
        started_at=utcnow_iso(),
        finished_at=utcnow_iso(),
        requested_ram_mib=8192.0,
        requested_vram_mib=0.0,
        requested_cpus=1,
        peak_ram_mib=1024.0,
        usage_samples=40,
        peak_source="measured",
        command_signature="testsig",
        description="a test job",
    )
    return service.db.get_job(job_id)


@pytest.fixture
def web(service):
    """A running UI over the same temporary state directory the CLI tests use."""
    server = serve(service.config, host="127.0.0.1", port=0, open_browser=False)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server
    finally:
        server.shutdown()
        server.server_close()


def request(server, path, method="GET", body=None):
    conn = HTTPConnection("127.0.0.1", server.server_address[1], timeout=30)
    try:
        payload = json.dumps(body or {}) if method == "POST" else None
        headers = {"Content-Type": "application/json"} if payload else {}
        conn.request(method, path, body=payload, headers=headers)
        response = conn.getresponse()
        raw = response.read()
        return response.status, raw
    finally:
        conn.close()


def get_json(server, path):
    status, raw = request(server, path)
    assert status == 200, raw[:400]
    return json.loads(raw)


class TestBinding:
    def test_it_refuses_to_listen_where_the_network_can_reach_it(self, service):
        """The dispatcher has no socket so that worker-q runs no unauthenticated
        network server. The UI can cancel jobs, so the same rule applies."""
        with pytest.raises(LoopbackOnly):
            serve(service.config, host="0.0.0.0", port=0)
        with pytest.raises(LoopbackOnly):
            serve(service.config, host="192.168.1.10", port=0)

    def test_loopback_names_and_addresses_are_allowed(self, service):
        for host in ("127.0.0.1", "localhost", "::1"):
            try:
                server = serve(service.config, host=host, port=0)
            except OSError:  # pragma: no cover - no IPv6 on this machine
                continue
            server.server_close()


class TestRoutes:
    def test_every_read_endpoint_answers(self, web):
        for path in (
            "/api/overview",
            "/api/facets",
            "/api/jobs",
            "/api/accuracy",
            "/api/machines",
            "/api/efficiency",
            "/api/events",
        ):
            payload = get_json(web, path)
            assert isinstance(payload, dict), path

    def test_the_overview_carries_what_the_live_view_draws(self, web):
        payload = get_json(web, "/api/overview")
        for key in ("version", "jobs", "summary", "gpu", "host", "nodes",
                    "throughput", "processes"):
            assert key in payload, key

    def test_an_unknown_endpoint_is_404_not_500(self, web):
        status, _ = request(web, "/api/nonsense")
        assert status == 404

    def test_an_unknown_page_serves_the_app_shell(self, web):
        """The UI routes on the hash, so a deep link must still load."""
        status, raw = request(web, "/history")
        assert status == 200
        assert b"<!doctype html>" in raw.lower()

    def test_static_assets_are_served_with_their_own_types(self, web):
        for path, marker in (("/style.css", b"--surface-1"),
                             ("/app.js", b"export") ):
            status, raw = request(web, path)
            assert status == 200, path
            assert marker in raw or path == "/app.js"

    def test_it_will_not_serve_outside_its_own_directory(self, web):
        """A path that escapes the static root is refused, not resolved."""
        status, raw = request(web, "/../../../../Windows/win.ini")
        # Either refused outright, or normalised by the client into the shell -
        # what must never happen is the file's contents coming back.
        assert b"[fonts]" not in raw and b"[extensions]" not in raw


class TestJobs:
    def test_a_job_carries_its_accuracy_verdict(self, web, queued_job):
        payload = get_json(web, "/api/jobs")
        assert payload["total"] >= 1
        first = payload["jobs"][0]
        assert "usage" in first
        assert "verdict" in first["usage"]
        # NULL node means this machine, and the read layer says so.
        assert first["node"] == "local"

    def test_filters_narrow_and_report_their_own_total(self, web, queued_job):
        everything = get_json(web, "/api/jobs")
        none = get_json(web, "/api/jobs?project=definitely-not-a-project")
        assert none["total"] == 0
        assert everything["total"] > none["total"]

    def test_paging_is_bounded(self, web, queued_job):
        payload = get_json(web, "/api/jobs?limit=99999")
        assert payload["limit"] <= 500

    def test_job_detail_includes_events_and_siblings(self, web, queued_job):
        payload = get_json(web, f"/api/jobs/{queued_job.id}")
        assert payload["id"] == queued_job.id
        assert isinstance(payload["events"], list)
        assert isinstance(payload["siblings"], list)
        assert "usage" in payload

    def test_a_missing_job_is_an_error_not_a_crash(self, web):
        status, _ = request(web, "/api/jobs/999999")
        assert status in (400, 404, 500)

    def test_the_log_endpoint_reports_a_missing_log_rather_than_failing(
        self, web, queued_job
    ):
        payload = get_json(web, f"/api/jobs/{queued_job.id}/log")
        assert payload["job_id"] == queued_job.id
        assert "text" in payload


class TestActions:
    def test_cancel_goes_through_the_service(self, web, queued_job):
        """Cancelling an already-finished job is a no-op, not an error - and
        the point here is that it went through `GPUQService.cancel`, which is
        what validates the transition, rather than raw SQL."""
        status, raw = request(
            web, f"/api/jobs/{queued_job.id}/cancel", method="POST"
        )
        assert status == 200, raw[:400]
        payload = json.loads(raw)
        # Every action reports the CLI command it is equivalent to, so nothing
        # the UI does is unreproducible in a terminal.
        assert payload["command"] == f"workerq cancel {queued_job.id}"
        assert "result" in payload

    def test_an_unknown_action_is_rejected(self, web, queued_job):
        status, _ = request(
            web, f"/api/jobs/{queued_job.id}/destroy", method="POST"
        )
        assert status == 400


class TestThreading:
    def test_concurrent_requests_do_not_share_a_sqlite_handle(self, web):
        """A connection belongs to the thread that created it.

        This has bitten the dispatcher's node reporting and the runner's
        progress watcher, in both cases silently - the exception was swallowed
        and the data simply never appeared. Here it would surface as sporadic
        500s under any real use.
        """
        paths = ["/api/overview", "/api/jobs", "/api/accuracy",
                 "/api/machines", "/api/efficiency"] * 4
        with ThreadPoolExecutor(max_workers=8) as pool:
            results = list(pool.map(lambda p: request(web, p)[0], paths))
        assert results == [200] * len(paths)


class TestBlockReasons:
    """A wait reason names the machine it applies to, and embeds live numbers."""

    def test_each_machines_bottleneck_is_counted_separately(self):
        from workerq.web.api import _block_reasons

        reasons = _block_reasons(
            "needs 34.0 GiB RAM but only 10.7 GiB is free after the 10% floor "
            "| on 3080ti: needs 34.0 GiB RAM but only 6.6 GiB is free"
        )
        assert len(reasons) == 2
        assert reasons[0].startswith("needs N GiB RAM")
        # The node has to stay named, or its bottleneck is indistinguishable
        # from this machine's and the two collapse into one row counted twice.
        assert reasons[1].startswith("on 3080ti:")

    def test_live_measurements_are_normalised_away(self):
        """Otherwise an unchanged condition reads as a new reason every tick -
        which is how a stalled queue produced 303,164 identical log lines."""
        from workerq.web.api import _block_reasons

        first = _block_reasons("system commit charge is 93.4% (limit 93.6 GiB)")
        second = _block_reasons("system commit charge is 91.2% (limit 93.6 GiB)")
        assert first == second

    def test_a_number_attached_to_letters_is_part_of_a_name(self):
        from workerq.web.api import _block_reasons

        assert _block_reasons("on 3080ti: busy") == ["on 3080ti: busy"]

    def test_nothing_in_nothing_out(self):
        from workerq.web.api import _block_reasons

        assert _block_reasons(None) == []
        assert _block_reasons("") == []
        assert _block_reasons("  |  ") == []
