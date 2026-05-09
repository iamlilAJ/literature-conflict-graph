#!/usr/bin/env python3
"""One-shot setup for the aigraph GitHub Projects v2 board.

Creates a user-level GitHub Project named "aigraph" with custom fields
(Stream, Priority, ETA, Blocker) and seeds it with the current task
state from work-status.md / TaskList.

Idempotent — re-run if you mess up; existing project + issues are
detected and updated rather than duplicated.

Requirements (must already be installed locally):
    - gh CLI (`brew install gh`)
    - gh authed (`gh auth login`) with scopes: repo, project
    - Run from anywhere; this script does NOT touch the working directory

Usage:
    python3 scripts/setup_github_project.py \
        --owner iamlilAJ \
        --repo literature-conflict-graph

After running, the script prints the Project URL. Open it in browser
or in the GitHub mobile app.

Re-run policy:
    Safe to re-run. Will:
    - Skip project creation if "aigraph" already exists for owner.
    - Skip field creation if field with same name exists.
    - Skip issue creation if issue with the same title already exists
      in the repo (matched on exact title prefix `[S<N>]`).
    - Skip view creation if view name already in project.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from typing import Any


PROJECT_TITLE = "aigraph"

STREAMS = [
    ("S1.Intern-Atlas",    "#FFA500"),
    ("S2.Validation",      "#4A90E2"),
    ("S3.OMC-Talent",      "#9013FE"),
    ("S4.aigraph-Core",    "#7ED321"),
    ("S5.Infra",           "#9B9B9B"),
]

PRIORITIES = ["P0", "P1", "P2"]


# Each task: title, body, stream, priority, status, eta, blocker, labels
TASKS: list[dict[str, Any]] = [
    # --- Active / Blocked / Next-up ---
    {
        "title": "[S1] commit + push controlled-validation-design.md amendments",
        "body": (
            "5-min wrap-up. `git commit -am \"docs: amend "
            "controlled-validation-design for Intern-Atlas pivot (closes "
            "#15)\"` and push. Closes the loop on the May 8 pivot session."
        ),
        "stream": "S1.Intern-Atlas",
        "priority": "P0",
        "status": "Todo",
        "eta": "5 min",
        "blocker": "",
        "labels": ["stream/s1-intern-atlas", "type/docs"],
    },
    {
        "title": "[S2] Draft docs/validation_preregistration_v1.md",
        "body": (
            "Lift H1-H5 from controlled-validation-design.md §11 into a "
            "standalone preregistration doc. MUST be committed BEFORE any "
            "predictions are locked, otherwise the experiment loses "
            "statistical credibility."
        ),
        "stream": "S2.Validation",
        "priority": "P0",
        "status": "Todo",
        "eta": "30 min",
        "blocker": "",
        "labels": ["stream/s2-validation", "type/docs"],
    },
    {
        "title": "[S1] corpus.intern_atlas_loader",
        "body": (
            "Add aigraph.corpus.intern_atlas_loader module that reads the "
            "local HF Intern-Atlas papers parquet and emits aigraph Paper "
            "models. Gated by env `AIGRAPH_USE_INTERN_ATLAS=1`. Keep "
            "S2/OpenAlex fetch path as fallback. Unit test schema mapping. "
            "Required for S2 prediction-pipeline runs."
        ),
        "stream": "S1.Intern-Atlas",
        "priority": "P1",
        "status": "Todo",
        "eta": "half day",
        "blocker": "",
        "labels": ["stream/s1-intern-atlas", "type/code"],
    },
    {
        "title": "[S2] Add prediction_year_cutoff guard to graph.build_graph",
        "body": (
            "Hard gate against look-ahead leakage (controlled-validation-"
            "design.md §8.3). `build_graph(prediction_year_cutoff: int | "
            "None = None)` — when set, drops edges whose source paper year > "
            "cutoff. Add unit test asserting no surviving edge violates the "
            "cutoff. Default `None` preserves existing behavior."
        ),
        "stream": "S2.Validation",
        "priority": "P1",
        "status": "Todo",
        "eta": "small (1-2 hours)",
        "blocker": "",
        "labels": ["stream/s2-validation", "type/code"],
    },
    {
        "title": "[S1] POC: query /api/eval against aigraph fixture",
        "body": (
            "Pick 30 hypotheses from existing aigraph fixture, send to "
            "Intern-Atlas /api/eval, compare 5-dim scores against aigraph "
            "influence_phase1, report ρ_aigraph vs ρ_intern_atlas. Decides "
            "whether the influence-prediction story is a marketed feature "
            "or supporting detail."
        ),
        "stream": "S1.Intern-Atlas",
        "priority": "P1",
        "status": "Blocked",
        "eta": "1-2 hours after unblocked",
        "blocker": "API waitlist. Apply at https://intern-atlas.opendatalab.org.cn",
        "labels": ["stream/s1-intern-atlas", "type/poc", "blocked"],
    },
    {
        "title": "[S5] User: run cleanup_branches.sh",
        "body": (
            "Blocked on user shell. Sandbox can't `git push origin "
            "--delete`. Script committed at repo root. Run: "
            "`cd <repo> && ./cleanup_branches.sh` then verify only "
            "origin/main + origin/gh-pages remain."
        ),
        "stream": "S5.Infra",
        "priority": "P2",
        "status": "Blocked",
        "eta": "5 min user time",
        "blocker": "User shell action required",
        "labels": ["stream/s5-infra", "blocked"],
    },
    # --- Parked ---
    {
        "title": "[S4] Phase 2 structural_impact dimension",
        "body": (
            "5th influence dimension (structural impact via graph repair "
            "Δ_h). Parked until Phase 1 4-dim validates on real cohort, "
            "because Phase 2 design choices need validation findings as "
            "input."
        ),
        "stream": "S4.aigraph-Core",
        "priority": "P2",
        "status": "Parked",
        "eta": "1-2 weeks when active",
        "blocker": "Phase 1 validation must complete first",
        "labels": ["stream/s4-core", "parked"],
    },
    {
        "title": "[S4] Investigate ρ_grounding_depth = -0.166 anomaly (PR #14)",
        "body": (
            "Backtest from PR #14 showed ρ_grounding_depth = -0.166. May "
            "resolve naturally on new Intern-Atlas cohort — wait for new "
            "data before spending cycles."
        ),
        "stream": "S4.aigraph-Core",
        "priority": "P2",
        "status": "Parked",
        "eta": "after Phase 1 cohort run",
        "blocker": "Wait for new ρ measurements",
        "labels": ["stream/s4-core", "parked", "investigation"],
    },
    {
        "title": "[S4] Top-10 precision improvement (currently 20%)",
        "body": (
            "Phase 1 top-10 precision is 20% on fixture. Wait for new ρ on "
            "Intern-Atlas cohort before tuning — cohort change may shift "
            "this number significantly."
        ),
        "stream": "S4.aigraph-Core",
        "priority": "P2",
        "status": "Parked",
        "eta": "after Phase 1 cohort run",
        "blocker": "Wait for new precision measurements",
        "labels": ["stream/s4-core", "parked"],
    },
    {
        "title": "[S3] OMC talent: declare Intern-Atlas as soft-dependency",
        "body": (
            "literature-researcher-talent should consume Intern-Atlas as "
            "preferred data source with OpenAlex/S2 fallback. README + "
            "schema adapter. Cross-repo work; lower priority than S1+S2 "
            "critical path."
        ),
        "stream": "S3.OMC-Talent",
        "priority": "P2",
        "status": "Parked",
        "eta": "1 day, after S1+S2 stabilize",
        "blocker": "S1+S2 critical path",
        "labels": ["stream/s3-omc", "parked", "cross-repo"],
    },
]


def run(cmd: list[str], capture: bool = True, check: bool = True) -> str:
    """Run a shell command, return stdout."""
    result = subprocess.run(
        cmd,
        capture_output=capture,
        text=True,
        check=False,
    )
    if check and result.returncode != 0:
        sys.stderr.write(f"\n!! command failed: {' '.join(cmd)}\n")
        sys.stderr.write(f"   stderr: {result.stderr}\n")
        raise SystemExit(result.returncode)
    return result.stdout.strip() if capture else ""


def gh_json(args: list[str]) -> Any:
    """Run a gh subcommand that emits JSON, parse and return."""
    out = run(args)
    if not out:
        return None
    return json.loads(out)


def ensure_gh_ready() -> None:
    if shutil.which("gh") is None:
        sys.exit(
            "gh CLI not found. Install: brew install gh, then gh auth login"
        )
    status = subprocess.run(
        ["gh", "auth", "status"], capture_output=True, text=True
    )
    if status.returncode != 0:
        sys.exit(
            "gh not authed. Run: gh auth login --scopes 'repo,project'"
        )


def find_project(owner: str, title: str) -> dict | None:
    out = gh_json([
        "gh", "project", "list",
        "--owner", owner,
        "--format", "json",
        "--limit", "200",
    ])
    if not out:
        return None
    for proj in out.get("projects", []):
        if proj.get("title") == title:
            return proj
    return None


def create_project(owner: str, title: str) -> dict:
    print(f"-- creating project '{title}' for {owner}")
    out = gh_json([
        "gh", "project", "create",
        "--owner", owner,
        "--title", title,
        "--format", "json",
    ])
    return out


def list_fields(owner: str, project_number: int) -> list[dict]:
    out = gh_json([
        "gh", "project", "field-list",
        str(project_number),
        "--owner", owner,
        "--format", "json",
        "--limit", "100",
    ])
    return out.get("fields", []) if out else []


def ensure_field(
    owner: str,
    project_number: int,
    name: str,
    data_type: str,
    options: list[str] | None = None,
) -> dict:
    """Ensure a custom field exists. Return its dict (incl id)."""
    fields = list_fields(owner, project_number)
    for f in fields:
        if f.get("name") == name:
            return f
    print(f"-- creating field '{name}' ({data_type})")
    cmd = [
        "gh", "project", "field-create",
        str(project_number),
        "--owner", owner,
        "--name", name,
        "--data-type", data_type,
        "--format", "json",
    ]
    if options:
        cmd += ["--single-select-options", ",".join(options)]
    return gh_json(cmd)


def find_issue(owner: str, repo: str, title: str) -> dict | None:
    """Look up an issue by exact title in the repo."""
    out = gh_json([
        "gh", "issue", "list",
        "--repo", f"{owner}/{repo}",
        "--state", "all",
        "--search", f'in:title "{title}"',
        "--json", "number,title,url,id",
        "--limit", "20",
    ])
    if not out:
        return None
    for issue in out:
        if issue.get("title") == title:
            return issue
    return None


def ensure_label(owner: str, repo: str, name: str, color: str = "ededed") -> None:
    existing = gh_json([
        "gh", "label", "list",
        "--repo", f"{owner}/{repo}",
        "--json", "name",
        "--limit", "200",
    ])
    if existing and any(l["name"] == name for l in existing):
        return
    run([
        "gh", "label", "create", name,
        "--repo", f"{owner}/{repo}",
        "--color", color,
    ], check=False)  # tolerate race


def create_issue(
    owner: str,
    repo: str,
    title: str,
    body: str,
    labels: list[str],
) -> dict:
    print(f"-- creating issue: {title}")
    for label in labels:
        ensure_label(owner, repo, label)
    cmd = [
        "gh", "issue", "create",
        "--repo", f"{owner}/{repo}",
        "--title", title,
        "--body", body,
    ]
    for label in labels:
        cmd += ["--label", label]
    out = run(cmd)
    # gh issue create prints the URL on stdout
    url = out.strip().splitlines()[-1] if out else ""
    return find_issue(owner, repo, title) or {"url": url, "title": title}


def add_item_to_project(owner: str, project_number: int, issue_url: str) -> str:
    out = gh_json([
        "gh", "project", "item-add",
        str(project_number),
        "--owner", owner,
        "--url", issue_url,
        "--format", "json",
    ])
    return out.get("id", "")


def edit_item_field(
    owner: str,
    project_id: str,
    item_id: str,
    field_id: str,
    *,
    text: str | None = None,
    single_select_option_id: str | None = None,
) -> None:
    cmd = [
        "gh", "project", "item-edit",
        "--id", item_id,
        "--project-id", project_id,
        "--field-id", field_id,
    ]
    if text is not None:
        cmd += ["--text", text]
    elif single_select_option_id is not None:
        cmd += ["--single-select-option-id", single_select_option_id]
    else:
        return
    run(cmd, check=False)


def get_project_full(owner: str, project_number: int) -> dict:
    """Pull the full project record incl. id and field option ids via GraphQL."""
    query = """
    query($login: String!, $number: Int!) {
      user(login: $login) {
        projectV2(number: $number) {
          id
          title
          url
          fields(first: 50) {
            nodes {
              ... on ProjectV2FieldCommon { id name dataType }
              ... on ProjectV2SingleSelectField {
                id name dataType
                options { id name }
              }
            }
          }
        }
      }
    }
    """
    out = gh_json([
        "gh", "api", "graphql",
        "-f", f"query={query}",
        "-F", f"login={owner}",
        "-F", f"number={project_number}",
    ])
    return out["data"]["user"]["projectV2"]


def setup_views(owner: str, project_number: int) -> None:
    """Note: gh CLI v2 does not yet support saved-view creation via project
    subcommand. Saved views are created via the web UI or GraphQL on
    ProjectV2View. We skip programmatic view creation and let the user
    add views in the GitHub web UI as a one-time follow-up."""
    print(
        "-- skipping saved-view creation (gh CLI does not yet support; "
        "create 3 views in web UI: 'By Stream' (table, group=Stream), "
        "'Kanban' (board, columns=Status), 'Today' (filter "
        "Status=Todo|InProgress sort by Priority))"
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--owner", required=True, help="GitHub username (e.g. iamlilAJ)")
    ap.add_argument("--repo",  required=True, help="Repo name (e.g. literature-conflict-graph)")
    args = ap.parse_args()

    ensure_gh_ready()

    owner = args.owner
    repo = args.repo

    # 1. Project
    proj = find_project(owner, PROJECT_TITLE)
    if not proj:
        proj = create_project(owner, PROJECT_TITLE)
    project_number = int(proj["number"])
    print(f"-- project #{project_number}: {proj.get('url')}")

    # 2. Custom fields
    ensure_field(owner, project_number, "Stream", "SINGLE_SELECT",
                 options=[s for s, _ in STREAMS])
    ensure_field(owner, project_number, "Priority", "SINGLE_SELECT",
                 options=PRIORITIES)
    ensure_field(owner, project_number, "ETA", "TEXT")
    ensure_field(owner, project_number, "Blocker", "TEXT")

    # 3. Pull full project + field option ids
    full = get_project_full(owner, project_number)
    project_id = full["id"]
    field_index = {}
    option_index = {}  # (field_name, option_name) -> option_id
    for f in full["fields"]["nodes"]:
        if not f:
            continue
        field_index[f["name"]] = f["id"]
        for opt in f.get("options", []) or []:
            option_index[(f["name"], opt["name"])] = opt["id"]

    # The default Status field uses "Todo / In Progress / Done" option names.
    # We add "Blocked" and "Parked" if missing — see web UI to add manually
    # (gh field-create can't extend an existing system field).

    # 4. Issues + project items
    for task in TASKS:
        existing = find_issue(owner, repo, task["title"])
        if existing:
            print(f"-- exists: {task['title']}")
            issue = existing
        else:
            issue = create_issue(
                owner, repo, task["title"], task["body"], task["labels"]
            )
        item_id = add_item_to_project(owner, project_number, issue["url"])
        if not item_id:
            continue

        # Set custom field values
        # Stream
        opt_id = option_index.get(("Stream", task["stream"]))
        if opt_id:
            edit_item_field(owner, project_id, item_id,
                            field_index["Stream"],
                            single_select_option_id=opt_id)
        # Priority
        opt_id = option_index.get(("Priority", task["priority"]))
        if opt_id:
            edit_item_field(owner, project_id, item_id,
                            field_index["Priority"],
                            single_select_option_id=opt_id)
        # ETA / Blocker (text)
        if task.get("eta"):
            edit_item_field(owner, project_id, item_id,
                            field_index["ETA"], text=task["eta"])
        if task.get("blocker"):
            edit_item_field(owner, project_id, item_id,
                            field_index["Blocker"], text=task["blocker"])

        # Status — only set if the desired option exists. Default GitHub
        # Status has Todo / In Progress / Done. We need user to add
        # "Blocked" + "Parked" via web UI for those to set successfully.
        desired_status = task.get("status")
        if desired_status and "Status" in field_index:
            opt_id = option_index.get(("Status", desired_status))
            if opt_id:
                edit_item_field(owner, project_id, item_id,
                                field_index["Status"],
                                single_select_option_id=opt_id)
            else:
                print(
                    f"   ! Status '{desired_status}' not in project — "
                    f"add it via web UI then re-run this script"
                )

    # 5. Views
    setup_views(owner, project_number)

    print()
    print("=" * 60)
    print(f"DONE. Open: {proj.get('url')}")
    print("=" * 60)
    print()
    print("Manual follow-ups (gh CLI doesn't support these yet):")
    print(" 1. In web UI, add custom Status options 'Blocked' + 'Parked'")
    print(" 2. Set Status on each item per the task definitions in this script")
    print(" 3. Create 3 saved views: 'By Stream' / 'Kanban' / 'Today'")


if __name__ == "__main__":
    main()
