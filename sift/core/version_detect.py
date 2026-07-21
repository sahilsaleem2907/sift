"""Detect a repo's declared target runtime version per language, for LLM grounding.

The per-file reviewer otherwise hallucinates the runtime version from its prior (e.g.
assuming an old Python and flagging a valid 3.13 API as missing). Detectors read only
small declaration files (pyproject.toml, package.json, go.mod, ...) via an injected
sync reader, so they are source-agnostic: the LLM path feeds a GitHub-API-backed reader
(clone-free), pyright feeds a clone-backed reader. Pure/sync → one source of truth,
trivially unit-testable.
"""
import logging
import re
import tomllib
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

logger = logging.getLogger(__name__)

Reader = Callable[[str], Optional[str]]


@dataclass(frozen=True)
class RuntimeTarget:
    language: str   # "python", "typescript", "go", "ruby", "java"
    summary: str    # prompt-ready, e.g. "Python 3.13 (repo's declared minimum)"


def _min_version_from_spec(spec: str) -> Optional[str]:
    """Minimum 'X.Y' from a version spec (e.g. '>=3.11,<3.14' -> '3.11')."""
    if not spec:
        return None
    m = re.search(r">=\s*(\d+)\.(\d+)", spec)
    if m:
        return f"{m.group(1)}.{m.group(2)}"
    m = re.search(r"~=\s*(\d+)\.(\d+)", spec) or re.search(r"==\s*(\d+)\.(\d+)", spec)
    if m:
        return f"{m.group(1)}.{m.group(2)}"
    m = re.search(r"(\d+)\.(\d+)", spec)
    return f"{m.group(1)}.{m.group(2)}" if m else None


class LanguageVersionDetector(ABC):
    language: str = ""
    exts: tuple = ()

    def applies_to(self, path: str) -> bool:
        return path.replace("\\", "/").lower().endswith(self.exts)

    @abstractmethod
    def files(self) -> tuple:
        """Declaration files this detector reads (so callers can pre-fetch them)."""

    @abstractmethod
    def detect(self, read: Reader) -> Optional[RuntimeTarget]:
        """Return the target for this language, or None if undeclared."""


class PythonVersionDetector(LanguageVersionDetector):
    language = "python"
    exts = (".py", ".pyi")

    def files(self) -> tuple:
        return ("pyproject.toml", "setup.cfg", "setup.py", ".python-version")

    def detect(self, read: Reader) -> Optional[RuntimeTarget]:
        pyproject = read("pyproject.toml")
        if pyproject:
            try:
                data = tomllib.loads(pyproject)
                proj = data.get("project")
                if isinstance(proj, dict):
                    v = _min_version_from_spec(str(proj.get("requires-python") or ""))
                    if v:
                        return RuntimeTarget(self.language, f"Python {v} (minimum of requires-python)")
                tool = data.get("tool") if isinstance(data.get("tool"), dict) else {}
                pr = tool.get("pyright") if isinstance(tool.get("pyright"), dict) else {}
                if pr.get("pythonVersion"):
                    return RuntimeTarget(self.language, f"Python {pr['pythonVersion']} (tool.pyright)")
                mp = tool.get("mypy") if isinstance(tool.get("mypy"), dict) else {}
                if mp.get("python_version"):
                    return RuntimeTarget(self.language, f"Python {mp['python_version']} (tool.mypy)")
            except Exception as e:
                logger.debug("pyproject.toml parse failed: %s", e)
        for name in ("setup.cfg", "setup.py"):
            text = read(name)
            if text:
                m = re.search(r"python_requires\s*=\s*['\"]?([^'\"\n]+)", text)
                if m:
                    v = _min_version_from_spec(m.group(1))
                    if v:
                        return RuntimeTarget(self.language, f"Python {v} (python_requires)")
        pv = read(".python-version")
        if pv:
            m = re.search(r"(\d+)\.(\d+)", pv)
            if m:
                return RuntimeTarget(self.language, f"Python {m.group(1)}.{m.group(2)} (.python-version)")
        return None


class NodeTsVersionDetector(LanguageVersionDetector):
    language = "typescript"
    exts = (".js", ".jsx", ".ts", ".tsx", ".mjs", ".cjs")

    def files(self) -> tuple:
        return ("package.json", "tsconfig.json")

    def detect(self, read: Reader) -> Optional[RuntimeTarget]:
        import json as _json
        parts: List[str] = []
        pkg = read("package.json")
        if pkg:
            try:
                engines = (_json.loads(pkg).get("engines") or {})
                if isinstance(engines, dict) and engines.get("node"):
                    parts.append(f"Node {engines['node']} (package.json engines)")
            except Exception as e:
                logger.debug("package.json parse failed: %s", e)
        tsconfig = read("tsconfig.json")
        if tsconfig:
            # tsconfig may contain comments/trailing commas; regex the target field.
            m = re.search(r'"target"\s*:\s*"([^"]+)"', tsconfig)
            if m:
                parts.append(f"TS target {m.group(1)} (tsconfig)")
        if parts:
            return RuntimeTarget(self.language, ", ".join(parts))
        return None


class GoVersionDetector(LanguageVersionDetector):
    language = "go"
    exts = (".go",)

    def files(self) -> tuple:
        return ("go.mod",)

    def detect(self, read: Reader) -> Optional[RuntimeTarget]:
        gomod = read("go.mod")
        if gomod:
            m = re.search(r"^\s*go\s+(\d+)\.(\d+)(?:\.\d+)?", gomod, re.MULTILINE)
            if m:
                return RuntimeTarget(self.language, f"Go {m.group(1)}.{m.group(2)} (go.mod)")
        return None


class RubyVersionDetector(LanguageVersionDetector):
    language = "ruby"
    exts = (".rb",)

    def files(self) -> tuple:
        return (".ruby-version", "Gemfile")

    def detect(self, read: Reader) -> Optional[RuntimeTarget]:
        rv = read(".ruby-version")
        if rv:
            m = re.search(r"(\d+)\.(\d+)(?:\.\d+)?", rv)
            if m:
                return RuntimeTarget(self.language, f"Ruby {m.group(1)}.{m.group(2)} (.ruby-version)")
        gemfile = read("Gemfile")
        if gemfile:
            m = re.search(r"^\s*ruby\s+['\"](\d+)\.(\d+)", gemfile, re.MULTILINE)
            if m:
                return RuntimeTarget(self.language, f"Ruby {m.group(1)}.{m.group(2)} (Gemfile)")
        return None


class JavaVersionDetector(LanguageVersionDetector):
    language = "java"
    exts = (".java",)

    def files(self) -> tuple:
        return ("pom.xml", "build.gradle", "build.gradle.kts")

    def detect(self, read: Reader) -> Optional[RuntimeTarget]:
        pom = read("pom.xml")
        if pom:
            m = (re.search(r"<maven\.compiler\.release>\s*(\d+)", pom)
                 or re.search(r"<maven\.compiler\.source>\s*(\d+)", pom)
                 or re.search(r"<release>\s*(\d+)\s*</release>", pom))
            if m:
                return RuntimeTarget(self.language, f"Java {m.group(1)} (pom.xml)")
        for name in ("build.gradle", "build.gradle.kts"):
            g = read(name)
            if g:
                m = re.search(r"sourceCompatibility\s*=?\s*['\"]?(?:JavaVersion\.VERSION_)?(\d+)", g)
                if m:
                    return RuntimeTarget(self.language, f"Java {m.group(1)} (build.gradle)")
        return None


DETECTORS: List[LanguageVersionDetector] = [
    PythonVersionDetector(),
    NodeTsVersionDetector(),
    GoVersionDetector(),
    RubyVersionDetector(),
    JavaVersionDetector(),
]

# Union of every declaration file the detectors read — callers pre-fetch these.
CANDIDATE_FILES: frozenset = frozenset(f for d in DETECTORS for f in d.files())


def detect_targets(read: Reader) -> Dict[str, RuntimeTarget]:
    """Run every detector against the reader; return language -> RuntimeTarget for those found."""
    out: Dict[str, RuntimeTarget] = {}
    for d in DETECTORS:
        try:
            t = d.detect(read)
        except Exception as e:
            logger.debug("version detector %s failed: %s", d.language, e)
            t = None
        if t:
            out[t.language] = t
    return out


def target_for_path(path: str, targets: Dict[str, RuntimeTarget]) -> Optional[RuntimeTarget]:
    """Pick the RuntimeTarget for the file's language, or None."""
    for d in DETECTORS:
        if d.applies_to(path):
            return targets.get(d.language)
    return None
