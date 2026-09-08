"""Extension host: agent lifecycle hooks and skill loading/injection."""

from .extension import AgentExtension, ExtensionContext
from .skill import Skill, SkillScript, Reference, inject, load_skill

__all__ = [
    "AgentExtension",
    "ExtensionContext",
    "Skill",
    "SkillScript",
    "Reference",
    "load_skill",
    "inject",
]
