---
name: marimo
description: Use when working with marimo notebooks in the zero-to-hero repo, especially for requests to open, run, restart, stop, or update marimo for lesson directories such as lesson-01 and lesson-02. This skill makes Codex use the lesson's local uv environment and launch marimo from the lesson folder instead of using a global marimo install.
---

# Zero To Hero Marimo

## Overview

This skill standardizes how marimo is managed in this repository. Always run marimo from the specific lesson directory with `uv run` so the notebook uses that lesson's pinned dependencies and local `.venv`.

## When To Use

Use this skill when the user asks to:

- Open or edit a marimo notebook in this repo
- Start, restart, or stop marimo for `lesson-01`, `lesson-02`, or similar lesson folders
- Update marimo for a lesson
- Run marimo in a way that must pick up `uv`, the lesson's `pyproject.toml`, or the lesson's `.venv`

## Rules

- Never start marimo from the repo root when the target notebook lives in a lesson directory.
- Never prefer the global `marimo` binary for this repo if the lesson is managed by `uv`.
- Set the shell working directory to the lesson directory first, then run commands there.
- Prefer `uv run marimo ...` over activating virtualenvs manually.
- If the user asks to update marimo, update the lesson-managed environment, not the global install.

## Workflow

1. Identify the lesson directory that owns the notebook, for example `lesson-02`.
2. Confirm `pyproject.toml` or `uv.lock` in that lesson if needed.
3. Run commands with `workdir` set to that lesson directory.
4. Use `uv sync` before launch when dependencies may be missing or the user asked for an update.
5. Launch with `uv run marimo edit notebook.py` unless the user explicitly wants a different marimo subcommand.
6. If a prior marimo session was started, stop that session before relaunching.
7. Return the local URL from marimo startup logs.

## Command Patterns

From a lesson directory such as `lesson-02`:

```bash
uv sync
uv run marimo --version
uv run marimo edit notebook.py
```

To verify the Python interpreter being used:

```bash
uv run python -c 'import sys; print(sys.executable)'
```

Expected result: the interpreter should resolve inside the lesson directory's `.venv`.

## Examples

- "Open marimo with lesson-02."
- "Restart marimo for lesson-01 using the local env."
- "Update marimo in lesson-02 and run it locally."
- "Stop the current marimo session and relaunch from the lesson folder."

No extra resources are required for this skill.
