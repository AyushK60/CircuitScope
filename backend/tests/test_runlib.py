"""Unit tests for scripts/runlib.py, the pure helpers behind start/stop.ps1
(free-port choice, port probes, pids.json parsing). No model, no network."""

from __future__ import annotations

import importlib.util
import json
import socket
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location("runlib", _REPO / "scripts" / "runlib.py")
runlib = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(runlib)


# --- choose_free_port (pure) --------------------------------------------------

def test_choose_free_port_skips_busy_and_is_inclusive():
    assert runlib.choose_free_port(5173, 5180, busy=set()) == 5173
    assert runlib.choose_free_port(5173, 5180, busy={5173, 5174}) == 5175
    assert runlib.choose_free_port(5173, 5173, busy={5173}) is None


def test_choose_free_port_returns_none_when_all_busy():
    assert runlib.choose_free_port(10, 12, busy={10, 11, 12}) is None


# --- probe_port_free (real socket, bind-based, both loopback stacks) ----------

def _bind_ephemeral(family: int = socket.AF_INET) -> tuple[socket.socket, int]:
    sock = socket.socket(family, socket.SOCK_STREAM)
    host = "127.0.0.1" if family == socket.AF_INET else "::1"
    sock.bind((host, 0))
    port = sock.getsockname()[1]
    sock.listen(1)
    return sock, port


def test_probe_port_free_sees_a_bound_ipv4_port_as_taken():
    sock, port = _bind_ephemeral(socket.AF_INET)
    try:
        assert runlib.probe_port_free(port) is False
    finally:
        sock.close()


def test_probe_port_free_sees_an_ipv6_only_holder_as_taken():
    """Regression: vite binds 'localhost', which can resolve to ::1; a port
    held only on IPv6 must not be handed out (real failure seen on this
    machine with another project on [::1]:5173)."""
    sock, port = _bind_ephemeral(socket.AF_INET6)
    try:
        assert runlib.probe_port_free(port) is False
    finally:
        sock.close()


def test_probe_port_free_reports_an_unused_port_as_free():
    sock, port = _bind_ephemeral(socket.AF_INET)
    sock.close()
    # after close the OS may keep the port reserved briefly; retry on flake
    for _ in range(20):
        if runlib.probe_port_free(port):
            break
        import time

        time.sleep(0.05)
    assert runlib.probe_port_free(port) is True


def test_find_free_port_picks_the_first_bindable_port():
    sock, port = _bind_ephemeral(socket.AF_INET)
    try:
        assert runlib.find_free_port(port, port) is None  # the only candidate is taken
    finally:
        sock.close()


# --- port_busy (connect-based; used by stop.ps1) -------------------------------

def test_port_busy_true_for_a_listening_socket_false_for_closed():
    sock, port = _bind_ephemeral(socket.AF_INET)
    try:
        assert runlib.port_busy(port) is True
    finally:
        sock.close()
    assert runlib.port_busy(port, timeout=0.2) is False


def test_port_busy_true_for_an_ipv6_only_listener():
    sock, port = _bind_ephemeral(socket.AF_INET6)
    try:
        assert runlib.port_busy(port) is True
    finally:
        sock.close()


def test_port_busy_refuses_a_dead_socket_quickly():
    # bind, close, then connect: connection refused -> not busy (no timeout wait)
    sock, port = _bind_ephemeral(socket.AF_INET)
    sock.close()
    assert runlib.port_busy(port, timeout=2.0) is False


# --- validate_pids / load_pids --------------------------------------------------

VALID = {
    "started_utc": "2026-11-19T00:00:00Z",
    "repo_root": "E:\\CircuitScope",
    "backend": {"pid": 111, "command_line": "python -m backend.app.main", "url": "http://127.0.0.1:8000", "port": 8000},
    "frontend": {"pid": 222, "command_line": "node vite.js --port 5173 --strictPort", "url": "http://localhost:5173", "port": 5173},
}


def test_validate_pids_accepts_and_normalizes_a_valid_record():
    out = runlib.validate_pids(VALID)
    assert out["backend"] == {"pid": 111, "command_line": "python -m backend.app.main", "port": 8000}
    assert out["frontend"]["pid"] == 222


def test_validate_pids_tolerates_extra_top_level_keys():
    record = dict(VALID)
    record["extra_future_field"] = "whatever"
    assert runlib.validate_pids(record)["backend"]["pid"] == 111


@pytest.mark.parametrize(
    "mutate",
    [
        lambda r: r.pop("backend"),
        lambda r: r.pop("frontend"),
        lambda r: r["backend"].__setitem__("pid", "111"),
        lambda r: r["backend"].__setitem__("pid", 0),
        lambda r: r["backend"].__setitem__("pid", True),
        lambda r: r["backend"].__setitem__("command_line", ""),
        lambda r: r["frontend"].__setitem__("port", 70000),
        lambda r: r["frontend"].__setitem__("port", None),
    ],
)
def test_validate_pids_rejects_malformed_records(mutate):
    import copy

    record = copy.deepcopy(VALID)
    mutate(record)
    with pytest.raises(runlib.PidsFileError):
        runlib.validate_pids(record)


def test_validate_pids_rejects_a_non_dict_top_level():
    with pytest.raises(runlib.PidsFileError):
        runlib.validate_pids([VALID])


def test_load_pids_roundtrip_and_errors(tmp_path):
    path = tmp_path / "pids.json"
    path.write_text(json.dumps(VALID), encoding="utf-8")
    assert runlib.load_pids(path)["backend"]["pid"] == 111

    with pytest.raises(runlib.PidsFileError):
        runlib.load_pids(tmp_path / "missing.json")

    bad = tmp_path / "bad.json"
    bad.write_text("{ not json", encoding="utf-8")
    with pytest.raises(runlib.PidsFileError):
        runlib.load_pids(bad)


def test_pids_file_written_by_start_script_shape_is_accepted():
    """The record start.ps1 writes (ConvertTo-Json ordered dict) must validate."""
    raw = (
        '{"started_utc":"2026-11-19T12:00:00Z","repo_root":"E:\\\\CircuitScope",'
        '"backend":{"pid":1234,"command_line":"E:\\\\CircuitScope\\\\.venv\\\\Scripts\\\\python.exe -m backend.app.main",'
        '"url":"http://127.0.0.1:8000","port":8000,"log":"outputs\\\\run\\\\backend.out.log"},'
        '"frontend":{"pid":5678,"command_line":"C:\\\\Program Files\\\\nodejs\\\\node.exe E:\\\\CircuitScope\\\\frontend\\\\node_modules\\\\vite\\\\bin\\\\vite.js --port 5173 --strictPort",'
        '"url":"http://localhost:5173","port":5173,"log":"outputs\\\\run\\\\frontend.out.log"}}'
    )
    out = runlib.validate_pids(json.loads(raw))
    assert out["backend"]["port"] == 8000 and out["frontend"]["port"] == 5173


def test_load_pids_reads_utf8_bom_files(tmp_path):
    """Set-Content -Encoding UTF8 in Windows PowerShell writes a BOM; the
    stop script must still be able to read what start wrote."""
    path = tmp_path / "pids.json"
    path.write_bytes(b"\xef\xbb\xbf" + json.dumps(VALID).encode("utf-8"))
    assert runlib.load_pids(path)["frontend"]["pid"] == 222
