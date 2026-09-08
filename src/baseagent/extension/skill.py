"""Skill loading and system-prompt injection.

A *skill* is a directory containing a ``SKILL.md`` (an optional YAML frontmatter
plus a markdown body) and, optionally, a ``reference/`` subtree of template
files the agent must follow, plus ``scripts/`` that the harness may run.

Two concerns live here:

- :func:`load_skill` — parse a skill directory into a :class:`Skill` (metadata,
  body, declared scripts with their execution mode, reference templates).
- :func:`inject` — render a skill (its body + any referenced templates) into
  a prompt fragment. Template placeholders (``{user_id}``, ``{index}`` …) are
  kept verbatim: they are *format exemplars* for the agent, not values the
  harness must substitute.

The ``scripts`` metadata distinguishes two kinds of script:

- ``mode: auto``   — the system runs it itself (not exposed to the agent).
- ``mode: agent``  — the agent may view/invoke it as a tool.

gitmem currently only publishes ``auto`` scripts; ``agent`` is reserved for
future tool-exposing scripts.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

import yaml


@dataclass
class SkillScript:
    name: str
    mode: str = "auto"  # "auto" | "agent"
    path: str = ""      # resolved script path (may not exist; advisory)
    description: str = ""


@dataclass
class Reference:
    """One template file under the skill's ``reference/`` tree.

    ``path`` is expressed relative to the reference root (e.g. ``index.md`` or
    ``explicit/e_{index}.md``) so callers can render a tidy label.
    """

    rel_path: str
    text: str = ""


@dataclass
class Skill:
    name: str = ""
    description: str = ""
    body: str = ""
    scripts: list[SkillScript] = field(default_factory=list)
    reference_dir: str = ""
    references: list[Reference] = field(default_factory=list)


def _split_frontmatter(text: str) -> tuple[dict[str, Any], str]:
    """Split a leading ``---`` YAML block from the markdown body."""
    if not text.startswith("---"):
        return {}, text
    lines = text.splitlines()
    # Find the closing `---` after the opening one.
    if len(lines) < 2:
        return {}, text
    for i in range(1, len(lines)):
        if lines[i].strip() == "---":
            raw = "\n".join(lines[1:i])
            body = "\n".join(lines[i + 1 :]).lstrip("\n")
            try:
                meta = yaml.safe_load(raw) or {}
            except yaml.YAMLError:
                meta = {}
            return (meta if isinstance(meta, dict) else {}), body
    return {}, text


def load_skill(skill_dir: str) -> Skill:
    """Load a skill directory into a :class:`Skill`.

    A skill directory has ``SKILL.md`` at its root; optional ``reference/``
    template files are collected recursively (sorted by relative path so the
    injected prompt is deterministic).
    """
    skill_dir = os.path.abspath(os.path.expanduser(skill_dir))
    md_path = os.path.join(skill_dir, "SKILL.md")
    if not os.path.isfile(md_path):
        raise FileNotFoundError(f"skill directory has no SKILL.md: {skill_dir}")

    with open(md_path, "r", encoding="utf-8") as f:
        text = f.read()

    meta, body = _split_frontmatter(text)
    name = str(meta.get("name") or os.path.basename(skill_dir))
    description = str(meta.get("description") or "")

    scripts: list[SkillScript] = []
    for spec in meta.get("scripts") or []:
        if isinstance(spec, str):
            scripts.append(SkillScript(name=spec))
        elif isinstance(spec, dict):
            scripts.append(
                SkillScript(
                    name=str(spec.get("name") or ""),
                    mode=str(spec.get("mode") or "auto"),
                    path=str(spec.get("path") or ""),
                    description=str(spec.get("description") or ""),
                )
            )

    reference_dir = str(meta.get("reference") or "")
    if reference_dir and not os.path.isabs(reference_dir):
        reference_dir = os.path.join(skill_dir, reference_dir)
    reference_dir = os.path.abspath(reference_dir) if reference_dir else ""

    refs = _collect_references(reference_dir)
    return Skill(
        name=name,
        description=description,
        body=body,
        scripts=scripts,
        reference_dir=reference_dir,
        references=refs,
    )


def _collect_references(reference_dir: str) -> list[Reference]:
    if not reference_dir or not os.path.isdir(reference_dir):
        return []
    out: list[Reference] = []
    for root, dirs, files in os.walk(reference_dir):
        # Deterministic order.
        dirs.sort()
        for fname in sorted(files):
            p = os.path.join(root, fname)
            try:
                with open(p, "r", encoding="utf-8") as f:
                    text = f.read()
            except OSError:
                continue
            rel = os.path.relpath(p, reference_dir)
            out.append(Reference(rel_path=rel, text=text))
    return out


def _script_lines(skill: Skill) -> str:
    if not skill.scripts:
        return ""
    lines = ["## Scripts"]
    for s in skill.scripts:
        mode = s.mode or "auto"
        desc = f" — {s.description}" if s.description else ""
        lines.append(f"- `{s.name}` ({mode}, system-run){desc}")
    return "\n".join(lines)


def inject(skill: Skill) -> str:
    """Render a skill into a prompt fragment.

    The fragment is the SKILL.md body (title/description folded in), followed
    by the reference templates as verbatim markdown blocks. Placeholders in the
    templates are intentionally left as-is.

    Before rendering, any mode: auto scripts that transform the Skill are
    applied (e.g., skill_fill for location-aware template injection).
    """
    # Apply skill_fill if present in scripts
    skill_transformed = skill
    for script in skill.scripts:
        if script.name == "skill_fill" and script.mode == "auto":
            skill_transformed = _apply_skill_fill(skill_transformed)
            break

    parts: list[str] = []
    if skill_transformed.description:
        parts.append(f"# {skill_transformed.name}\n{skill_transformed.description}")
    if skill_transformed.body.strip():
        parts.append(skill_transformed.body.strip())
    script_block = _script_lines(skill_transformed)
    if script_block:
        parts.append(script_block)
    if skill_transformed.references:
        parts.append("## Reference templates (follow these exactly)")
        for ref in skill_transformed.references:
            parts.append(f"### {ref.rel_path}\n\n```markdown\n{ref.text.rstrip()}\n```")
    return "\n\n".join(p for p in parts if p.strip())


def _apply_skill_fill(skill: Skill) -> Skill:
    """Apply skill_fill transformation if the module is available.

    Dynamically imports fill_skill from the skill's scripts package and applies
    location-aware template injection. Falls back to the original skill if the
    module cannot be imported.
    """
    try:
        # Try importing from the skill's scripts package
        # The skill scripts are expected to be importable via gitmem_scripts
        from gitmem_scripts.skill_fill import fill_skill
        return fill_skill(skill)
    except ImportError:
        # If skill_fill module doesn't exist, return the skill unchanged
        return skill


__all__ = [
    "Skill",
    "SkillScript",
    "Reference",
    "load_skill",
    "inject",
]
