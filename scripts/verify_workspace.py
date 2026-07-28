#!/usr/bin/env python3
"""Verify the evolving data repository and pinned method repositories."""

from __future__ import annotations

import os
import subprocess
import tempfile
import tomllib
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def git(path: Path, *args: str, env: dict[str, str] | None = None) -> str:
    process = subprocess.run(
        ["git", "-C", str(path), *args],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )
    return process.stdout.strip()


def resolve(relative: str) -> Path:
    return (ROOT / relative).resolve()


def verify_branch(name: str, path: Path, expected: str) -> None:
    actual = git(path, "branch", "--show-current")
    if actual != expected:
        raise AssertionError(f"{name}: expected branch {expected}, got {actual}")


def verify_clone(method: dict[str, Any], path: Path) -> None:
    actual = git(path, "rev-parse", "HEAD")
    if actual != method["revision"]:
        raise AssertionError(
            f"{method['display_name']}: expected {method['revision']}, got {actual}"
        )
    if git(path, "status", "--porcelain"):
        raise AssertionError(f"{method['display_name']}: upstream worktree is dirty")


def verify_archive_tree(method: dict[str, Any], path: Path) -> None:
    actual_revision = git(path, "rev-parse", "HEAD")
    if actual_revision != method["local_revision"]:
        raise AssertionError(
            f"{method['display_name']}: expected local snapshot "
            f"{method['local_revision']}, got {actual_revision}"
        )
    if git(path, "status", "--porcelain"):
        raise AssertionError(f"{method['display_name']}: snapshot worktree is dirty")
    with tempfile.TemporaryDirectory(prefix="benchmark-git-index.") as temp_name:
        index_path = Path(temp_name) / "index"
        env = dict(os.environ)
        env["GIT_INDEX_FILE"] = str(index_path)
        git(path, "add", "-f", "--all", env=env)
        actual_tree = git(path, "write-tree", env=env)
    if actual_tree != method["tree"]:
        raise AssertionError(
            f"{method['display_name']}: expected tree {method['tree']}, got {actual_tree}"
        )


def main() -> None:
    manifest_path = ROOT / "manifests" / "repositories.toml"
    with manifest_path.open("rb") as handle:
        manifest = tomllib.load(handle)

    data = manifest["data"]
    data_path = resolve(data["path"])
    data_root = Path(git(data_path, "rev-parse", "--show-toplevel")).resolve()
    if data_root != data_path:
        raise AssertionError(
            f"data: expected an isolated repository at {data_path}, "
            f"got Git root {data_root}"
        )
    verify_branch("data", data_path, data["branch"])
    data_revision = git(data_path, "rev-parse", "HEAD")
    if git(data_path, "status", "--porcelain"):
        raise AssertionError("data repository worktree is dirty")
    print(f"OK data {data_revision[:12]}")

    for method in manifest["methods"]:
        path = resolve(method["path"])
        if not path.is_dir():
            raise FileNotFoundError(f"Missing {method['display_name']}: {path}")
        verify_branch(method["display_name"], path, method["branch"])
        remote = git(path, "remote", "get-url", "upstream")
        if remote != method["repository"]:
            raise AssertionError(
                f"{method['display_name']}: unexpected upstream remote {remote}"
            )
        for entrypoint in method["entrypoints"]:
            if not (path / entrypoint).is_file():
                raise FileNotFoundError(
                    f"{method['display_name']}: missing entrypoint {entrypoint}"
                )
        if method["acquisition"] == "git_clone":
            verify_clone(method, path)
        elif method["acquisition"] == "verified_archive":
            verify_archive_tree(method, path)
        else:
            raise ValueError(f"Unknown acquisition mode: {method['acquisition']}")
        print(
            f"OK {method['display_name']} {method['revision'][:12]} "
            f"({method['integration_status']})"
        )


if __name__ == "__main__":
    main()
