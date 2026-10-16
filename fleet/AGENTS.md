# Fleet GitOps (raelldottin fork only)

The Tachikoma fleet runs Hermes from this fork's `fleet-stable` branch: upstream `main` plus the
fork's unmerged fixes. Git is the only way to change what runs where.

## Source of truth

- `fleet/environments/<env>.toml` on `fleet-stable` is the desired state of `<env>`: the pinned
  `commit` (full SHA) and optional `tag` (must point at it), the deploy checkout (`source_dir`), its
  `HERMES_HOME`, and `deploy = "enabled" | "disabled"`. Production defaults to `"disabled"`.
- Only what is **merged on `fleet-stable`** deploys. `reconcile.py` reads manifests from the fetched
  branch, never from a working tree, so a local or unmerged edit cannot release anything.
- `promote_from` lists the environments a commit must run in first. Production promotes from
  canary: `check` requires its commit to be at or behind canary's, and `apply` refuses unless the
  canary's audit log records a successful deployment of that exact commit.
- Production `deploy = "enabled"` is invalid without `approved_by = "<human>"` in the same PR.

## Workflow

1. **Change code:** branch from `fleet-stable`, add the fix with 1–2 regression tests proven red
   on the base, open a PR into `fleet-stable`. `.github/workflows/fleet-validate.yml` must pass.
2. **Release to canary:** a PR that sets `canary.toml` `commit` to the merged SHA. After merge, in
   the canary checkout: `python fleet/reconcile.py plan canary`, then `apply canary`, then restart
   the canary gateway. Record the canary report in the PR.
3. **Promote to production:** a PR that sets `production.toml` `commit` to a SHA the canary ran and
   flips `deploy` to `"enabled"` with `approved_by`. Only the operator merges it. Then
   `plan`/`apply production` and the operator restarts the production gateway.
4. **Roll back:** revert the manifest PR (or pin the previous SHA), merge, `apply`. `apply` copies
   every kanban DB and `config.yaml` to `<hermes_home>/fleet/backups/<stamp>/` before moving the
   checkout; stop the gateway and restore that copy if the newer code wrote state the older one
   cannot read.

## Audit trail

Every `apply`, refused or successful, appends one JSON line to `<hermes_home>/fleet/deployments.jsonl`:
environment, commit, tag, previous commit, approver, backup path, outcome, error, timestamp. The
manifest PR records who approved the release; the log records what actually happened on the host.

## Rules for agents

- **Never deploy by hand.** No `git checkout`/`pull`/`stash` in a deploy checkout
  (`~/.hermes/hermes-agent` is production's); only `reconcile.py apply` moves one, and it refuses a
  dirty tree or a disabled environment.
- **Never edit a manifest to unblock yourself.** `deploy = "disabled"` and `approved_by` are human
  decisions; never set them, and never merge a manifest PR.
- **`apply` never restarts services.** Restarting the production gateway, `launchctl`, and
  `hermes update` on a fleet host need explicit operator approval every time; unattended
  `hermes update` would switch a parked `fleet-stable` checkout back to upstream `main`.
- **Non-production `hermes_home` lives outside `~/.hermes`** (`check` enforces it): any home under
  it shares the production kanban root. A canary also uses its own ports and test credentials only.
- **Keep the patch set small.** When upstream merges an equivalent fix, rebase `fleet-stable`, drop
  the local patch once its regression test passes on upstream alone, and release that as above.
- **Keep fork-only files in `fleet/`** (plus `tests/fleet/` and `fleet-validate.yml`) so upstream
  rebases stay conflict-free; don't edit upstream files for fleet concerns.

## Secrets

`fleet-validate.yml` runs untrusted PR code, so it has read-only `contents` permission, uses no
secrets or environments, and never deploys. Deploying runs `reconcile.py` on the fleet host, pulling
from GitHub; nothing in CI holds credentials for a fleet host. Never add a `pull_request_target`
trigger or deploy job to a workflow that checks out PR code.

## Testing

- `python fleet/reconcile.py check` — manifests valid, commits on the branch, promotion order held.
- `scripts/run_tests.sh tests/fleet/` — reconciler invariants: disabled and dirty targets never
  move, unmerged edits can't deploy, approval, canary gate, tag identity, backups, audit records.
- `fleet-validate.yml` runs both plus `scripts/check` against the upstream base and the kanban
  suites; a failing run blocks the merge, so nothing unvalidated reaches a manifest. Before
  promoting to production, also run the full kanban suite on Linux and macOS against the upstream
  base and compare failures; a failure only on `fleet-stable` blocks promotion.
