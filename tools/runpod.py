#!/usr/bin/env python3
"""Lease-aware RunPod helpers for ChaosControl experiments.

Adapted from parameter-golf/tools/run_deepfloor_runpod.py.
Single GPU, auto-stop lease, rsync results back.

Usage:
    python tools/runpod.py create
    python tools/runpod.py start <pod_id>
    python tools/runpod.py extend <pod_id>
    python tools/runpod.py harvest-stop <pod_id>
    python tools/runpod.py lease-status
"""
from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import sys
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REMOTE_REPO_ROOT = Path("/workspace/chaoscontrol")
LEASE_STATE_ROOT = Path.home() / ".cache" / "chaoscontrol" / "runpod_leases"
DEFAULT_LEASE_MINUTES = 480  # 8 hours — enough for the full 70-config run
WATCH_POLL_SECONDS = 30
LEASE_SCHEMA_VERSION = 1


def run_json(cmd: list[str]) -> dict[str, Any] | list[Any]:
    result = subprocess.run(cmd, check=True, capture_output=True, text=True)
    return json.loads(result.stdout)


def run_passthrough(cmd: list[str]) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def ensure_local_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def epoch_now() -> int:
    return int(time.time())


def iso_utc(epoch_seconds: int) -> str:
    return datetime.fromtimestamp(epoch_seconds, tz=UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def owner_label(explicit_owner: str | None = None) -> str:
    if explicit_owner:
        return explicit_owner
    return f"{os.getenv('USER', 'unknown')}@{socket.gethostname()}:{os.getpid()}"


def ensure_lease_state_root() -> Path:
    LEASE_STATE_ROOT.mkdir(parents=True, exist_ok=True)
    return LEASE_STATE_ROOT


def lease_state_path(pod_id: str) -> Path:
    return LEASE_STATE_ROOT / f"{pod_id}.json"


def load_lease_state(pod_id: str) -> dict[str, Any] | None:
    path = lease_state_path(pod_id)
    if not path.exists():
        return None
    return json.loads(path.read_text())


def save_lease_state(state: dict[str, Any]) -> Path:
    ensure_lease_state_root()
    path = lease_state_path(str(state["pod_id"]))
    path.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n")
    return path


def clear_lease_state(pod_id: str) -> None:
    path = lease_state_path(pod_id)
    if path.exists():
        path.unlink()


def find_nested_value(payload: Any, candidate_keys: tuple[str, ...]) -> Any | None:
    if isinstance(payload, dict):
        for key in candidate_keys:
            if key in payload and payload[key] not in (None, ""):
                return payload[key]
        for value in payload.values():
            found = find_nested_value(value, candidate_keys)
            if found not in (None, ""):
                return found
    elif isinstance(payload, list):
        for item in payload:
            found = find_nested_value(item, candidate_keys)
            if found not in (None, ""):
                return found
    return None


def extract_pod_id(payload: dict[str, Any] | list[Any]) -> str:
    pod_id = find_nested_value(payload, ("id", "podId", "pod_id"))
    if not isinstance(pod_id, str) or not pod_id:
        raise RuntimeError(f"Unable to determine pod id from RunPod response: {payload}")
    return pod_id


def get_pod_details(pod_id: str) -> dict[str, Any]:
    payload = run_json(["runpodctl", "pod", "get", pod_id, "-o", "json"])
    if not isinstance(payload, dict):
        raise RuntimeError(f"Unexpected pod detail payload for {pod_id}: {payload}")
    return payload


def get_ssh_info(pod_id: str) -> dict[str, Any]:
    payload = run_json(["runpodctl", "ssh", "info", pod_id, "-v", "-o", "json"])
    if not isinstance(payload, dict):
        raise RuntimeError(f"Unexpected ssh info payload for {pod_id}: {payload}")
    if payload.get("error"):
        raise RuntimeError(f"Unable to fetch ssh info for {pod_id}: {payload['error']}")
    return payload


def get_pod_name_with_fallback(pod_id: str, fallback_payload: dict[str, Any] | list[Any] | None = None) -> str:
    try:
        details = get_pod_details(pod_id)
        name = details.get("name") or find_nested_value(fallback_payload or {}, ("name",))
        return str(name or pod_id)
    except Exception:
        return str(find_nested_value(fallback_payload or {}, ("name",)) or pod_id)


def new_lease_record(lease_minutes: int, owner: str, reason: str, now_epoch: int) -> dict[str, Any]:
    return {
        "lease_id": uuid.uuid4().hex,
        "owner": owner,
        "reason": reason,
        "created_at": now_epoch,
        "expires_at": now_epoch + (lease_minutes * 60),
    }


def append_lease_to_state(
    existing_state: dict[str, Any] | None,
    *,
    pod_id: str,
    pod_name: str | None,
    lease_record: dict[str, Any],
    watch_token: str,
    now_epoch: int,
) -> dict[str, Any]:
    leases = list(existing_state.get("leases", [])) if existing_state else []
    leases.append(lease_record)
    state = {
        "schema_version": LEASE_SCHEMA_VERSION,
        "pod_id": pod_id,
        "pod_name": pod_name or (existing_state or {}).get("pod_name") or pod_id,
        "watch_token": watch_token,
        "updated_at": now_epoch,
        "leases": leases,
    }
    state["created_at"] = (existing_state or {}).get("created_at", now_epoch)
    return state


def active_leases(state: dict[str, Any], now_epoch: int) -> list[dict[str, Any]]:
    return [
        lease for lease in state.get("leases", [])
        if isinstance(lease, dict) and int(lease.get("expires_at", 0)) > now_epoch
    ]


def format_lease_summary(state: dict[str, Any], now_epoch: int) -> str:
    active = active_leases(state, now_epoch)
    total = len([l for l in state.get("leases", []) if isinstance(l, dict)])
    next_expiry = max((int(l["expires_at"]) for l in active), default=None)
    owners = sorted({str(l.get("owner", "?")) for l in active})
    parts = [f"pod={state.get('pod_id')}", f"name={state.get('pod_name')}",
             f"active={len(active)}", f"total={total}"]
    if next_expiry:
        parts.append(f"expires={iso_utc(next_expiry)}")
    if owners:
        parts.append(f"owners={','.join(owners)}")
    return " ".join(parts)


def spawn_watchdog(pod_id: str, watch_token: str) -> None:
    cmd = [sys.executable, str(Path(__file__).resolve()), "watch", pod_id, "--watch-token", watch_token]
    subprocess.Popen(
        cmd, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL, start_new_session=True, close_fds=True,
    )


def arm_lease(pod_id: str, *, pod_name: str | None, lease_minutes: int, owner: str, reason: str) -> dict[str, Any]:
    now_epoch = epoch_now()
    lease_record = new_lease_record(lease_minutes, owner, reason, now_epoch)
    existing_state = load_lease_state(pod_id)
    watch_token = uuid.uuid4().hex
    state = append_lease_to_state(
        existing_state, pod_id=pod_id, pod_name=pod_name,
        lease_record=lease_record, watch_token=watch_token, now_epoch=now_epoch,
    )
    save_lease_state(state)
    spawn_watchdog(pod_id, watch_token)
    print(f"armed lease for pod {pod_id} until {iso_utc(int(lease_record['expires_at']))} ({lease_minutes}m)")
    print(format_lease_summary(state, now_epoch))
    return state


def build_rsync_command(*, pod_id: str, remote_path: str, local_path: Path) -> list[str]:
    ssh_info = get_ssh_info(pod_id)
    host = str(ssh_info["ip"])
    port = str(ssh_info["port"])
    ssh_key = str(ssh_info["ssh_key"]["path"])
    ensure_local_parent(local_path)
    return [
        "rsync", "-az", "--progress", "-e",
        f"ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -i {ssh_key} -p {port}",
        f"root@{host}:{remote_path}",
        str(local_path),
    ]


def build_rsync_push_command(*, pod_id: str, local_path: Path, remote_path: str) -> list[str]:
    ssh_info = get_ssh_info(pod_id)
    host = str(ssh_info["ip"])
    port = str(ssh_info["port"])
    ssh_key = str(ssh_info["ssh_key"]["path"])
    return [
        "rsync", "-az", "--no-o", "--no-g", "--progress",
        "--exclude", ".git", "--exclude", ".claude", "--exclude", ".venv",
        "--exclude", ".DS_Store", "--exclude", "__pycache__",
        "--exclude", "*.pyc", "--exclude", "results/", "-e",
        f"ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -i {ssh_key} -p {port}",
        str(local_path) + "/",
        f"root@{host}:{remote_path}/",
    ]


def build_ssh_command(pod_id: str, remote_cmd: str) -> list[str]:
    ssh_info = get_ssh_info(pod_id)
    host = str(ssh_info["ip"])
    port = str(ssh_info["port"])
    ssh_key = str(ssh_info["ssh_key"]["path"])
    return [
        "ssh", "-o", "StrictHostKeyChecking=no", "-o", "UserKnownHostsFile=/dev/null",
        "-i", ssh_key, "-p", port, f"root@{host}", remote_cmd,
    ]


# --- Commands ---

def cmd_create(args: argparse.Namespace) -> None:
    name = args.name or f"chaoscontrol-{int(time.time())}"
    cmd = [
        "runpodctl", "pod", "create",
        "--template-id", args.template_id,
        "--gpu-id", args.gpu_id,
        "--gpu-count", str(args.gpu_count),
        "--name", name,
        "--ports", "22/tcp",
        "--cloud-type", args.cloud_type,
        "--container-disk-in-gb", str(args.container_disk_gb),
        "--volume-in-gb", str(args.volume_gb),
        "-o", "json",
    ]
    if args.public_ip:
        cmd.append("--public-ip")
    payload = run_json(cmd)
    pod_id = extract_pod_id(payload)
    arm_lease(
        pod_id,
        pod_name=get_pod_name_with_fallback(pod_id, fallback_payload=payload),
        lease_minutes=args.lease_minutes,
        owner=owner_label(args.owner),
        reason="create",
    )
    print(json.dumps(payload, indent=2))


def cmd_deploy(args: argparse.Namespace) -> None:
    """Push repo to pod; optionally run the broad bootstrap script."""
    pod_id = args.pod_id
    print("=== Pushing repo to pod ===")
    push_cmd = build_rsync_push_command(
        pod_id=pod_id,
        local_path=REPO_ROOT,
        remote_path=str(DEFAULT_REMOTE_REPO_ROOT),
    )
    run_passthrough(push_cmd)

    if not args.bootstrap:
        print("\n=== Skipping bootstrap ===")
        print("Use --bootstrap only when you intentionally want tools/pod_bootstrap.sh.")
        return

    print("\n=== Running legacy broad bootstrap (venv + smoke test + batch benchmark) ===")
    ssh_cmd = build_ssh_command(
        pod_id,
        "CHAOSCONTROL_ALLOW_BROAD_BOOTSTRAP=1 "
        f"bash {DEFAULT_REMOTE_REPO_ROOT}/tools/pod_bootstrap.sh",
    )
    run_passthrough(ssh_cmd)


def cmd_start(args: argparse.Namespace) -> None:
    run_passthrough(["runpodctl", "pod", "start", args.pod_id])
    arm_lease(
        args.pod_id,
        pod_name=get_pod_name_with_fallback(args.pod_id),
        lease_minutes=args.lease_minutes,
        owner=owner_label(args.owner),
        reason="start",
    )


def cmd_stop(args: argparse.Namespace) -> None:
    run_passthrough(["runpodctl", "pod", "stop", args.pod_id])
    clear_lease_state(args.pod_id)


def cmd_extend(args: argparse.Namespace) -> None:
    arm_lease(
        args.pod_id,
        pod_name=get_pod_name_with_fallback(args.pod_id),
        lease_minutes=args.lease_minutes,
        owner=owner_label(args.owner),
        reason="extend",
    )


def cmd_harvest(args: argparse.Namespace) -> None:
    """Pull results from pod."""
    local_path = REPO_ROOT / "experiments"
    remote_path = f"{DEFAULT_REMOTE_REPO_ROOT}/experiments/"
    cmd = build_rsync_command(pod_id=args.pod_id, remote_path=remote_path, local_path=local_path)
    run_passthrough(cmd)
    print(f"Results synced to {local_path}")


def cmd_harvest_stop(args: argparse.Namespace) -> None:
    cmd_harvest(args)
    run_passthrough(["runpodctl", "pod", "stop", args.pod_id])
    clear_lease_state(args.pod_id)


def cmd_lease_status(args: argparse.Namespace) -> None:
    now_epoch = epoch_now()
    if args.pod_id:
        state = load_lease_state(args.pod_id)
        if state is None:
            print(f"no lease state found for pod {args.pod_id}")
            return
        print(format_lease_summary(state, now_epoch))
        return
    if not LEASE_STATE_ROOT.exists():
        print("no tracked pod leases")
        return
    for path in sorted(LEASE_STATE_ROOT.glob("*.json")):
        state = json.loads(path.read_text())
        print(format_lease_summary(state, now_epoch))


def cmd_watch(args: argparse.Namespace) -> None:
    while True:
        state = load_lease_state(args.pod_id)
        if state is None:
            return
        if state.get("watch_token") != args.watch_token:
            return
        now_epoch = epoch_now()
        active = active_leases(state, now_epoch)
        if not active:
            try:
                run_passthrough(["runpodctl", "pod", "stop", args.pod_id])
            except subprocess.CalledProcessError:
                pass
            clear_lease_state(args.pod_id)
            return
        next_expiry = max(int(l["expires_at"]) for l in active)
        time.sleep(max(1, min(WATCH_POLL_SECONDS, next_expiry - now_epoch)))


def add_lease_args(parser: argparse.ArgumentParser, default_minutes: int = DEFAULT_LEASE_MINUTES) -> None:
    parser.add_argument("--lease-minutes", type=int, default=default_minutes)
    parser.add_argument("--owner")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="ChaosControl RunPod orchestration")
    sub = p.add_subparsers(dest="command", required=True)

    create = sub.add_parser("create", help="Create a GPU pod")
    create.add_argument("--template-id", default="y5cejece4j")
    create.add_argument("--gpu-id", default="NVIDIA A40 48GB")
    create.add_argument("--gpu-count", type=int, default=3)
    create.add_argument("--name")
    create.add_argument("--cloud-type", default="COMMUNITY")
    create.add_argument("--public-ip", action=argparse.BooleanOptionalAction, default=True)
    create.add_argument("--container-disk-gb", type=int, default=50)
    create.add_argument("--volume-gb", type=int, default=100)
    add_lease_args(create)
    create.set_defaults(handler=cmd_create)

    deploy = sub.add_parser("deploy", help="Push repo to pod")
    deploy.add_argument("pod_id")
    deploy.add_argument(
        "--bootstrap",
        action="store_true",
        help="also run tools/pod_bootstrap.sh after sync (legacy broad environment mutation)",
    )
    deploy.set_defaults(handler=cmd_deploy)

    start = sub.add_parser("start", help="Start a stopped pod")
    start.add_argument("pod_id")
    add_lease_args(start)
    start.set_defaults(handler=cmd_start)

    stop = sub.add_parser("stop", help="Stop a pod")
    stop.add_argument("pod_id")
    stop.set_defaults(handler=cmd_stop)

    extend = sub.add_parser("extend", help="Add lease time")
    extend.add_argument("pod_id")
    add_lease_args(extend, default_minutes=120)
    extend.set_defaults(handler=cmd_extend)

    harvest = sub.add_parser("harvest", help="Pull results from pod")
    harvest.add_argument("pod_id")
    harvest.set_defaults(handler=cmd_harvest)

    harvest_stop = sub.add_parser("harvest-stop", help="Pull results and stop pod")
    harvest_stop.add_argument("pod_id")
    harvest_stop.set_defaults(handler=cmd_harvest_stop)

    status = sub.add_parser("lease-status", help="Show lease status")
    status.add_argument("pod_id", nargs="?")
    status.set_defaults(handler=cmd_lease_status)

    watch = sub.add_parser("watch", help=argparse.SUPPRESS)
    watch.add_argument("pod_id")
    watch.add_argument("--watch-token", required=True)
    watch.set_defaults(handler=cmd_watch)

    return p


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.handler(args)


if __name__ == "__main__":
    main()
