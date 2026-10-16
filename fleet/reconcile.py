#!/usr/bin/env python3
"""GitOps reconciler for the Tachikoma Hermes fleet (raelldottin fork only).

Desired state is ``fleet/environments/<env>.toml`` as merged on ``fleet-stable``. A deploy
checkout pulls that branch, reads the manifest from the fetched commit (never from a working
tree, so an unmerged edit cannot deploy) and moves itself to the pinned commit.

    python fleet/reconcile.py check                 # validate manifests at HEAD (CI runs this)
    python fleet/reconcile.py plan  canary          # fetch, show drift for one environment
    python fleet/reconcile.py apply canary          # move the deploy checkout to the pinned commit

``plan``/``apply`` run inside the environment's own deploy checkout (its ``source_dir``).

``apply`` backs up the target's kanban/config state, moves the checkout, appends an audit record to
``<hermes_home>/fleet/deployments.jsonl`` (success or failure) and never restarts anything.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sqlite3
import subprocess
import sys
import time
import tomllib
from pathlib import Path

ENV_DIR = "fleet/environments"
BRANCH = "fleet-stable"
REQUIRED = ("environment", "commit", "source_dir", "hermes_home", "deploy")
DEPLOY_STATES = ("enabled", "disabled")
GIT_TIMEOUT_SECONDS = 300


class FleetError(Exception):
    pass


def live_home() -> Path:
    """Production's install root: fixed, whatever profile the reconciler itself runs under."""
    return Path.home() / ".hermes"  # health: allow HX001 -- production root, not the caller's profile


def _git(repo: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess:
    proc = subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True,
                          check=False, timeout=GIT_TIMEOUT_SECONDS)
    if check and proc.returncode != 0:
        raise FleetError(f"git {' '.join(args)}: {proc.stderr.strip()}")
    return proc


def _is_ancestor(repo: Path, older: str, newer: str) -> bool:
    return _git(repo, "merge-base", "--is-ancestor", older, newer, check=False).returncode == 0


def _under(path: Path, root: Path) -> bool:
    return path == root or root in path.parents


def _tag_commit(repo: Path, tag: str) -> str | None:
    proc = _git(repo, "rev-parse", "--verify", "--quiet", f"refs/tags/{tag}^{{commit}}", check=False)
    return proc.stdout.strip() or None


def load(repo: Path, ref: str) -> dict[str, dict]:
    """Every manifest as committed at ``ref``, keyed by environment name."""
    names = _git(repo, "ls-tree", "--name-only", f"{ref}:{ENV_DIR}").stdout.split()
    manifests = {}
    for name in sorted(n for n in names if n.endswith(".toml")):
        data = tomllib.loads(_git(repo, "show", f"{ref}:{ENV_DIR}/{name}").stdout)
        manifests[data.get("environment", name.removesuffix(".toml"))] = data
    return manifests


def validate(repo: Path, ref: str, manifests: dict[str, dict]) -> list[str]:
    errors = []
    for env, m in manifests.items():
        missing = [k for k in REQUIRED if k not in m]
        if missing:
            errors.append(f"{env}: missing {', '.join(missing)}")
            continue
        if m["deploy"] not in DEPLOY_STATES:
            errors.append(f"{env}: deploy must be one of {DEPLOY_STATES}, got {m['deploy']!r}")
        if not re.fullmatch(r"[0-9a-f]{40}", m["commit"]):
            errors.append(f"{env}: commit must be a full 40-character SHA")
        elif not _is_ancestor(repo, m["commit"], ref):
            errors.append(f"{env}: commit {m['commit'][:10]} is not on {ref}")
        elif "tag" in m and _tag_commit(repo, m["tag"]) != m["commit"]:
            errors.append(f"{env}: tag {m['tag']} does not point at commit {m['commit'][:10]}")
        if env == "production" and m["deploy"] == "enabled" and not m.get("approved_by"):
            errors.append(f"{env}: deploy = \"enabled\" needs approved_by (the human who approved it)")
        home = Path(m["hermes_home"]).expanduser()
        if env != "production" and _under(home, live_home()):
            # Any HERMES_HOME under ~/.hermes resolves the production kanban root.
            errors.append(f"{env}: hermes_home must be outside {live_home()}")
        for gate in m.get("promote_from", []):
            if gate not in manifests:
                errors.append(f"{env}: promote_from names unknown environment {gate!r}")
            elif "commit" in manifests[gate] and not _is_ancestor(repo, m["commit"], manifests[gate]["commit"]):
                errors.append(f"{env}: commit is not yet deployed to {gate} (promote through it first)")
    return errors


def _fetch_desired(checkout: Path, env: str, remote: str) -> tuple[dict, dict[str, dict]]:
    """Fetch fleet-stable into this deploy checkout and read ``env``'s manifest from it."""
    _git(checkout, "fetch", "--quiet", "--tags", remote, BRANCH)
    desired = load(checkout, "FETCH_HEAD")
    errors = validate(checkout, "FETCH_HEAD", desired)
    if errors:
        raise FleetError(f"manifests on {remote}/{BRANCH} are invalid:\n  " + "\n  ".join(errors))
    if env not in desired:
        raise FleetError(f"{env!r} is not defined on {remote}/{BRANCH}; known: {', '.join(desired)}")
    m = desired[env]
    if Path(m["source_dir"]).expanduser().resolve() != checkout.resolve():
        # Each environment owns one checkout; refuse to move some other one.
        raise FleetError(f"{env} deploys from {m['source_dir']}, not {checkout}")
    return m, desired


def plan(checkout: Path, env: str, remote: str) -> dict:
    m, desired = _fetch_desired(checkout, env, remote)
    current = _git(checkout, "rev-parse", "HEAD").stdout.strip()
    dirty = bool(_git(checkout, "status", "--porcelain").stdout.strip())
    return {"environment": env, "source_dir": str(checkout), "hermes_home": m["hermes_home"],
            "desired": m["commit"], "current": current, "in_sync": current == m["commit"],
            "dirty": dirty, "deploy": m["deploy"], "tag": m.get("tag"),
            "approved_by": m.get("approved_by"),
            "gates": {g: desired[g]["hermes_home"] for g in m.get("promote_from", [])}}


def _audit_log(hermes_home: str) -> Path:
    return Path(hermes_home).expanduser() / "fleet" / "deployments.jsonl"


def _record(hermes_home: str, entry: dict) -> None:
    log = _audit_log(hermes_home)
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry, sort_keys=True) + "\n")


def _succeeded_in(env_home: str, commit: str) -> bool:
    log = _audit_log(env_home)
    if not log.exists():
        return False
    return any(r.get("commit") == commit and r.get("outcome") == "success"
               for r in map(json.loads, log.read_text(encoding="utf-8").splitlines()))


def _backup_state(hermes_home: str, stamp: str) -> Path:
    """Consistent copies of every kanban DB plus config, so a rollback can restore state."""
    home = Path(hermes_home).expanduser()
    dest = home / "fleet" / "backups" / stamp
    dest.mkdir(parents=True)
    for db in [home / "kanban.db", *sorted((home / "kanban" / "boards").glob("*/kanban.db"))]:
        if db.exists():
            target = dest / db.relative_to(home)
            target.parent.mkdir(parents=True, exist_ok=True)
            with sqlite3.connect(db) as src, sqlite3.connect(target) as out:
                src.backup(out)
    if (home / "config.yaml").exists():
        shutil.copy2(home / "config.yaml", dest / "config.yaml")
    return dest


def apply(checkout: Path, env: str, remote: str) -> dict:
    p = plan(checkout, env, remote)
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    entry = {"environment": env, "commit": p["desired"], "tag": p["tag"], "previous": p["current"],
             "approved_by": p["approved_by"], "source_dir": p["source_dir"], "at": stamp}
    try:
        if p["deploy"] != "enabled":
            raise FleetError(f"{env} is disabled in {ENV_DIR}/{env}.toml; merge an approved PR "
                             'setting deploy = "enabled" to release it')
        if p["dirty"]:
            raise FleetError(f"{checkout} has uncommitted changes; refusing to move it")
        for gate, gate_home in p["gates"].items():
            if not _succeeded_in(gate_home, p["desired"]):
                raise FleetError(f"{p['desired'][:10]} has no successful {gate} deployment on record")
        if not p["in_sync"]:
            entry["backup"] = str(_backup_state(p["hermes_home"], stamp))
            _git(checkout, "checkout", "--quiet", "--detach", p["desired"])
    except FleetError as exc:
        _record(p["hermes_home"], {**entry, "outcome": "refused", "error": str(exc)})
        raise
    _record(p["hermes_home"], {**entry, "outcome": "success"})
    # ponytail: the restart is left to the operator; automate it once production is enabled.
    return {**p, **entry, "current": p["desired"], "in_sync": True, "outcome": "success",
            "next_step": f"restart the {env} gateway (HERMES_HOME={p['hermes_home']}) to load it"}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=("check", "plan", "apply"))
    parser.add_argument("environment", nargs="?")
    parser.add_argument("--remote", default="origin", help="remote holding fleet-stable (default: origin)")
    parser.add_argument("--checkout", type=Path, default=Path.cwd(),
                        help="deploy checkout to plan/apply (default: current directory)")
    args = parser.parse_args(argv)
    try:
        if args.command == "check":
            errors = validate(Path.cwd(), "HEAD", load(Path.cwd(), "HEAD"))
            for e in errors:
                print(f"fleet: {e}", file=sys.stderr)
            print("fleet manifests ok" if not errors else f"fleet: {len(errors)} error(s)")
            return 1 if errors else 0
        if not args.environment:
            parser.error(f"{args.command} needs an environment")
        result = (plan if args.command == "plan" else apply)(args.checkout, args.environment, args.remote)
        print(json.dumps(result, indent=2))
        return 0
    except FleetError as exc:
        print(f"fleet: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
