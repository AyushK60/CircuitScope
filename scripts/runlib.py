"""Pure helpers for the CircuitScope start/stop PowerShell scripts.

Standard library only. The PowerShell scripts (start.ps1, stop.ps1) call this
file's CLI so the logic below is unit-testable (backend/tests/test_runlib.py):

    python scripts/runlib.py choose-port 5173 5180   # prints a free port, or exit 3
    python scripts/runlib.py port-status 8000 5173   # prints "<port> busy|free", exit 4 if any busy
    python scripts/runlib.py load-pids outputs/run/pids.json  # prints validated JSON, or exit 5

Two different port probes with different purposes:
- probe_port_free() uses a bind() test on BOTH loopback stacks (IPv4 127.0.0.1
  and IPv6 ::1): "can something listen here on every stack localhost uses?".
  Used to choose the frontend port, so a port held on either stack (including
  TIME_WAIT residue) is skipped. Vite binds "localhost", which can resolve to
  either stack, so a port free on IPv4 but held on IPv6 must be rejected.
- port_busy() uses a TCP connect() test on both stacks: "is something listening
  here?". Used to confirm ports are free after stopping, where a bind test
  could give a false alarm from TIME_WAIT sockets left by the just-stopped
  server.
"""

from __future__ import annotations

import argparse
import json
import socket
import sys
from pathlib import Path

EXIT_NO_FREE_PORT = 3
EXIT_PORT_BUSY = 4
EXIT_BAD_PIDS = 5


class PidsFileError(ValueError):
    """pids.json is missing, unreadable or does not have the expected shape."""


def _bind_ok(sock_family: int, port: int, host: str) -> bool:
    sock = socket.socket(sock_family, socket.SOCK_STREAM)
    try:
        sock.bind((host, port))
        return True
    except OSError as e:
        # No IPv6 support at all -> treat the IPv6 check as passed; the IPv4
        # bind is then the only meaningful one. (Windows WSAEAFNOTSUPPORT =
        # 10047; BSD-ish errno values vary by platform, so also match the
        # message.)
        if sock_family == socket.AF_INET6 and (
            getattr(e, "winerror", None) == 10047
            or getattr(e, "errno", None) in (97, 102)
            or "address family" in str(e).lower()
        ):
            return True
        return False
    finally:
        sock.close()


def probe_port_free(port: int) -> bool:
    """True when we can bind this port on every loopback stack that exists.

    Vite binds "localhost", which may resolve to 127.0.0.1 or ::1, so a port
    held on only one stack is not safe to hand out.
    """
    if not _bind_ok(socket.AF_INET, port, "127.0.0.1"):
        return False
    return _bind_ok(socket.AF_INET6, port, "::1")


def choose_free_port(start_port: int, end_port: int, busy: set[int] | None = None) -> int | None:
    """First port in [start_port, end_port] not in `busy` (pure; unit-tested)."""
    busy = busy or set()
    for port in range(start_port, end_port + 1):
        if port not in busy:
            return port
    return None


def find_free_port(start_port: int, end_port: int) -> int | None:
    """First bindable port in the range, probing the real OS (None if all taken)."""
    for port in range(start_port, end_port + 1):
        if probe_port_free(port):
            return port
    return None


def _connect_ok(sock_family: int, host: str, port: int, timeout: float) -> bool:
    sock = socket.socket(sock_family, socket.SOCK_STREAM)
    sock.settimeout(timeout)
    try:
        sock.connect((host, port))
        return True
    except OSError:
        return False
    finally:
        sock.close()


def port_busy(port: int, timeout: float = 1.0) -> bool:
    """True when something is accepting connections on this port on either
    loopback stack (a listening server on ::1 alone still counts as busy)."""
    if _connect_ok(socket.AF_INET, "127.0.0.1", port, timeout):
        return True
    return _connect_ok(socket.AF_INET6, "::1", port, timeout)


def _validated_entry(name: str, entry: object) -> dict:
    if not isinstance(entry, dict):
        raise PidsFileError(f"pids.json: '{name}' entry is not an object")
    pid = entry.get("pid")
    if isinstance(pid, bool) or not isinstance(pid, int) or pid <= 0:
        raise PidsFileError(f"pids.json: '{name}.pid' must be a positive integer, got {pid!r}")
    command_line = entry.get("command_line")
    if not isinstance(command_line, str) or not command_line.strip():
        raise PidsFileError(f"pids.json: '{name}.command_line' must be a non-empty string")
    port = entry.get("port")
    if isinstance(port, bool) or not isinstance(port, int) or not (0 < port < 65536):
        raise PidsFileError(f"pids.json: '{name}.port' must be a valid port number, got {port!r}")
    return {"pid": pid, "command_line": command_line, "port": port}


def validate_pids(record: object) -> dict:
    """Validate a parsed pids.json structure; returns the canonical form.

    Raises PidsFileError with a plain message on any problem. Unknown extra
    top-level keys are tolerated (forward compatibility).
    """
    if not isinstance(record, dict):
        raise PidsFileError("pids.json: top level must be an object")
    out: dict = {}
    for name in ("backend", "frontend"):
        if name not in record:
            raise PidsFileError(f"pids.json: missing '{name}' entry")
        out[name] = _validated_entry(name, record[name])
    return out


def load_pids(path: str | Path) -> dict:
    """Read and validate a pids.json file."""
    p = Path(path)
    if not p.exists():
        raise PidsFileError(f"pids file not found: {p}")
    try:
        raw = p.read_text(encoding="utf-8-sig")
        parsed = json.loads(raw)
    except json.JSONDecodeError as e:
        raise PidsFileError(f"pids file {p} is not valid JSON: {e}") from e
    except OSError as e:
        raise PidsFileError(f"pids file {p} could not be read: {e}") from e
    return validate_pids(parsed)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Pure helpers for start/stop scripts.")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_choose = sub.add_parser("choose-port", help="print the first bindable port in a range")
    p_choose.add_argument("start", type=int)
    p_choose.add_argument("end", type=int)

    p_status = sub.add_parser("port-status", help="print '<port> busy|free' per port")
    p_status.add_argument("ports", nargs="+", type=int)

    p_pids = sub.add_parser("load-pids", help="validate a pids.json and print it")
    p_pids.add_argument("path")

    args = parser.parse_args(argv)

    if args.cmd == "choose-port":
        port = find_free_port(args.start, args.end)
        if port is None:
            print(f"no free port in [{args.start}, {args.end}]", file=sys.stderr)
            return EXIT_NO_FREE_PORT
        print(port)
        return 0

    if args.cmd == "port-status":
        any_busy = False
        for port in args.ports:
            busy = port_busy(port)
            print(f"{port} {'busy' if busy else 'free'}")
            any_busy = any_busy or busy
        return EXIT_PORT_BUSY if any_busy else 0

    if args.cmd == "load-pids":
        try:
            record = load_pids(args.path)
        except PidsFileError as e:
            print(str(e), file=sys.stderr)
            return EXIT_BAD_PIDS
        print(json.dumps(record, indent=2))
        return 0

    return 0  # unreachable: subparsers are required


if __name__ == "__main__":
    sys.exit(main())
