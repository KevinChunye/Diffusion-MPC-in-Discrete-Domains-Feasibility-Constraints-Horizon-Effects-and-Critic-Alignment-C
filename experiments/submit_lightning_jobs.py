from __future__ import annotations

import argparse
import datetime as dt
import os
import re
import shlex
import subprocess
from typing import Dict, List

import yaml


def _extract_commands(plan_path: str) -> List[str]:
    with open(plan_path, "r", encoding="utf-8") as f:
        plan = yaml.safe_load(f) or {}
    proposals = plan.get("proposals", [])
    cmds = []
    for p in proposals:
        cmd = str(p.get("command", "")).strip()
        if cmd:
            cmds.append(cmd)
    return cmds


def _job_id_from_output(text: str) -> str:
    patterns = [
        r"job[_\s-]*id[:=\s]+([A-Za-z0-9._-]+)",
        r"\bid[:=\s]+([A-Za-z0-9._-]+)",
        r"\bjob[-_][A-Za-z0-9._-]+",
    ]
    for pat in patterns:
        m = re.search(pat, text, flags=re.IGNORECASE)
        if m:
            return m.group(1) if m.lastindex else m.group(0)
    return ""


def _build_submit_cmd(job_command: str, args: argparse.Namespace) -> List[str]:
    cmd = ["lightning", "run", "job", f"--command={job_command}"]
    if args.org:
        cmd.append(f"--org={args.org}")
    if args.teamspace:
        cmd.append(f"--teamspace={args.teamspace}")
    if args.cluster:
        cmd.append(f"--cluster={args.cluster}")
    if args.machine:
        cmd.append(f"--machine={args.machine}")
    if args.gpu:
        cmd.append(f"--gpu={args.gpu}")
    return cmd


def _write_log(log_path: str, plan_path: str, rows: List[Dict[str, str]], dry_run: bool) -> None:
    os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)
    lines = []
    lines.append("# Lightning Submission Log")
    lines.append("")
    lines.append(f"- Timestamp: {dt.datetime.now().isoformat(timespec='seconds')}")
    lines.append(f"- Plan: `{plan_path}`")
    lines.append(f"- Dry run: `{dry_run}`")
    lines.append("")
    lines.append("| # | Status | Job ID | Submit Command | Planned Command |")
    lines.append("|---:|---|---|---|---|")
    for i, r in enumerate(rows, start=1):
        lines.append(
            "| {idx} | {status} | {job_id} | `{submit}` | `{planned}` |".format(
                idx=i,
                status=r.get("status", ""),
                job_id=r.get("job_id", ""),
                submit=r.get("submit_cmd", "").replace("|", "\\|"),
                planned=r.get("planned_cmd", "").replace("|", "\\|"),
            )
        )
    with open(log_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def main() -> None:
    p = argparse.ArgumentParser(description="Submit tuner plan commands to Lightning jobs.")
    p.add_argument("--plan_path", type=str, required=True)
    p.add_argument("--org", type=str, default="")
    p.add_argument("--teamspace", type=str, default="")
    p.add_argument("--cluster", type=str, default="")
    p.add_argument("--machine", type=str, default="")
    p.add_argument("--gpu", type=str, default="")
    p.add_argument("--dry_run", action="store_true")
    p.add_argument("--log_dir", type=str, default="runs/submissions")
    args = p.parse_args()

    planned_cmds = _extract_commands(args.plan_path)
    if not planned_cmds:
        raise RuntimeError(f"No runnable commands found in plan: {args.plan_path}")

    ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(args.log_dir, f"{ts}.md")
    rows: List[Dict[str, str]] = []

    for planned in planned_cmds:
        submit = _build_submit_cmd(planned, args)
        submit_str = shlex.join(submit)
        print(submit_str)

        if args.dry_run:
            rows.append(
                {
                    "status": "DRY_RUN",
                    "job_id": "",
                    "submit_cmd": submit_str,
                    "planned_cmd": planned,
                }
            )
            continue

        try:
            proc = subprocess.run(submit, text=True, capture_output=True, check=True)
            out = (proc.stdout or "") + "\n" + (proc.stderr or "")
            job_id = _job_id_from_output(out)
            rows.append(
                {
                    "status": "SUBMITTED",
                    "job_id": job_id,
                    "submit_cmd": submit_str,
                    "planned_cmd": planned,
                }
            )
        except subprocess.CalledProcessError as exc:
            out = (exc.stdout or "") + "\n" + (exc.stderr or "")
            rows.append(
                {
                    "status": "FAILED",
                    "job_id": _job_id_from_output(out),
                    "submit_cmd": submit_str,
                    "planned_cmd": planned,
                }
            )

    _write_log(log_path, args.plan_path, rows, args.dry_run)
    print(f"Submission log: {log_path}")


if __name__ == "__main__":
    main()
