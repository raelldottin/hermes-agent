"""GitOps reconciler: only manifests merged on fleet-stable can move a deploy checkout."""
import sqlite3
import subprocess
from pathlib import Path

import pytest

from fleet import reconcile


def _git(repo, *args):
    return subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True,
                          text=True).stdout.strip()


def _commit(repo, files, message):
    for rel, text in files.items():
        path = repo / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    _git(repo, "add", "-A")
    _git(repo, "-c", "user.name=t", "-c", "user.email=t@example.invalid", "commit", "-qm", message)
    return _git(repo, "rev-parse", "HEAD")


def _manifest(env, commit, source_dir, hermes_home, deploy, promote_from=(), extra=""):
    gates = f"promote_from = {list(promote_from)!r}\n".replace("'", '"') if promote_from else ""
    return (f'environment = "{env}"\ncommit = "{commit}"\nsource_dir = "{source_dir}"\n'
            f'hermes_home = "{hermes_home}"\ndeploy = "{deploy}"\n{gates}{extra}')


@pytest.fixture
def fleet(tmp_path):
    """An upstream fleet repo, plus a canary deploy checkout cloned from it."""
    origin = tmp_path / "origin"
    origin.mkdir()
    _git(origin, "init", "-q", "-b", "fleet-stable")
    base = _commit(origin, {"app.txt": "v1"}, "v1")
    newer = _commit(origin, {"app.txt": "v2"}, "v2")
    canary = tmp_path / "canary"
    subprocess.run(["git", "clone", "-q", "--branch", "fleet-stable", str(origin), str(canary)], check=True)
    _git(canary, "checkout", "-q", "--detach", base)

    prod = tmp_path / "prod"
    subprocess.run(["git", "clone", "-q", "--branch", "fleet-stable", str(origin), str(prod)], check=True)
    _git(prod, "checkout", "-q", "--detach", base)

    def publish(deploy="enabled", commit=newer, home=tmp_path / "canary-home", prod_commit=None,
                prod_deploy="disabled", prod_extra=""):
        files = {"fleet/environments/canary.toml": _manifest("canary", commit, canary, home, deploy)}
        if prod_commit:
            files["fleet/environments/production.toml"] = _manifest(
                "production", prod_commit, prod, tmp_path / "prod-home", prod_deploy, ["canary"], prod_extra)
        return _commit(origin, files, "manifests")

    return {"origin": origin, "canary": canary, "prod": prod, "base": base, "newer": newer,
            "home": tmp_path / "canary-home", "prod_home": tmp_path / "prod-home", "publish": publish}


def _audit(home):
    import json
    return [json.loads(line) for line in (home / "fleet/deployments.jsonl").read_text().splitlines()]


def test_apply_moves_checkout_to_merged_commit(fleet):
    fleet["publish"]()
    board = fleet["home"] / "kanban/boards/main/kanban.db"
    board.parent.mkdir(parents=True)
    with sqlite3.connect(board) as db:
        db.execute("create table t (x)")
    receipt = reconcile.apply(fleet["canary"], "canary", "origin")
    assert receipt["previous"] == fleet["base"]
    assert _git(fleet["canary"], "rev-parse", "HEAD") == fleet["newer"]
    assert (Path(receipt["backup"]) / "kanban/boards/main/kanban.db").exists()
    [record] = _audit(fleet["home"])
    assert (record["outcome"], record["commit"], record["previous"]) == ("success", fleet["newer"], fleet["base"])


def test_hold_and_dirty_checkout_never_move(fleet):
    fleet["publish"](deploy="disabled")
    with pytest.raises(reconcile.FleetError, match="disabled"):
        reconcile.apply(fleet["canary"], "canary", "origin")
    fleet["publish"]()
    (fleet["canary"] / "app.txt").write_text("local edit")
    with pytest.raises(reconcile.FleetError, match="uncommitted"):
        reconcile.apply(fleet["canary"], "canary", "origin")
    assert _git(fleet["canary"], "rev-parse", "HEAD") == fleet["base"]
    assert [r["outcome"] for r in _audit(fleet["home"])] == ["refused", "refused"]


def test_production_needs_approval_and_a_successful_canary(fleet):
    newer = fleet["newer"]
    fleet["publish"](prod_commit=newer, prod_deploy="enabled")
    errors = reconcile.validate(fleet["origin"], "HEAD", reconcile.load(fleet["origin"], "HEAD"))
    assert errors == ['production: deploy = "enabled" needs approved_by (the human who approved it)']
    fleet["publish"](prod_commit=newer, prod_deploy="enabled", prod_extra='approved_by = "operator"\n')
    with pytest.raises(reconcile.FleetError, match="no successful canary deployment"):
        reconcile.apply(fleet["prod"], "production", "origin")
    reconcile.apply(fleet["canary"], "canary", "origin")
    receipt = reconcile.apply(fleet["prod"], "production", "origin")
    assert receipt["approved_by"] == "operator"
    assert _git(fleet["prod"], "rev-parse", "HEAD") == newer


def test_unmerged_local_manifest_edit_cannot_deploy(fleet):
    """Desired state comes from the fetched branch, not the deploy checkout's files."""
    fleet["publish"](deploy="disabled")
    reconcile.plan(fleet["canary"], "canary", "origin")
    _git(fleet["canary"], "checkout", "-q", "--detach", "FETCH_HEAD")
    manifest = fleet["canary"] / "fleet/environments/canary.toml"
    manifest.write_text(manifest.read_text().replace('"disabled"', '"enabled"'))
    _commit(fleet["canary"], {}, "local release attempt")
    with pytest.raises(reconcile.FleetError, match="disabled"):
        reconcile.apply(fleet["canary"], "canary", "origin")


def test_check_rejects_unsafe_manifests(fleet):
    origin = fleet["origin"]
    fleet["publish"](home=reconcile.live_home() / "profiles" / "canary", prod_commit=fleet["newer"])
    errors = reconcile.validate(origin, "HEAD", reconcile.load(origin, "HEAD"))
    assert any("outside" in e for e in errors)
    # Production may not run a commit the canary has not run.
    fleet["publish"](commit=fleet["base"], prod_commit=fleet["newer"])
    errors = reconcile.validate(origin, "HEAD", reconcile.load(origin, "HEAD"))
    assert errors == ["production: commit is not yet deployed to canary (promote through it first)"]
    _git(origin, "tag", "release-1", fleet["base"])
    fleet["publish"](prod_commit=fleet["newer"], prod_extra='tag = "release-1"\n')
    errors = reconcile.validate(origin, "HEAD", reconcile.load(origin, "HEAD"))
    assert errors == [f"production: tag release-1 does not point at commit {fleet['newer'][:10]}"]
