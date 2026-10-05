"""Independent core/optional dependency and reviewed CVE policies."""
import re
import tomllib
from pathlib import Path

from packaging.requirements import Requirement
from packaging.version import Version

REPO_ROOT = Path(__file__).resolve().parents[1]


def _canonical(name):
    return re.sub(r"[-_.]+", "-", name).lower()


def _exact_pins(specs):
    """Yield (name, spec) for every requirement pinned to one exact version."""
    for spec in specs:
        requirement = Requirement(spec)
        clauses = list(requirement.specifier)
        if len(clauses) != 1 or clauses[0].operator != "==":
            continue
        if "*" in clauses[0].version:
            continue
        yield requirement.name, spec


def _exclude_newer_exemptions():
    metadata = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    entries = metadata["tool"]["uv"].get("exclude-newer-package", {})
    return metadata, {_canonical(name) for name in entries}


def test_test_dependencies_are_group_only_in_manifest_and_lock():
    manifest = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    lock = tomllib.loads((REPO_ROOT / "uv.lock").read_text(encoding="utf-8"))
    hermes = next(package for package in lock["package"] if package["name"] == manifest["project"]["name"])
    assert manifest["tool"]["uv"]["default-groups"] == []
    assert "dev" in manifest["dependency-groups"]
    assert "dev" not in manifest["project"]["optional-dependencies"]
    assert "dev" in hermes["dev-dependencies"]
    assert "dev" not in hermes.get("optional-dependencies", {})


def test_core_and_optional_speech_dependencies():
    project = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    core = {Requirement(dep).name for dep in project["dependencies"]}
    assert "packaging" in core  # Runtime code imports it directly, not transitively.
    assert "faster-whisper" not in core
    assert "faster-whisper" in {
        Requirement(dep).name for dep in project["optional-dependencies"]["stt-whisper"]
    }


def test_starlette_server_pins_and_lock_exclude_cve_2026_48710():
    # BadHost's reviewed fixed boundary is independent of today's exact pin.
    floor = Version("1.0.1")
    metadata = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    lock = tomllib.loads((REPO_ROOT / "uv.lock").read_text(encoding="utf-8"))
    found = set()
    for extra, specs in metadata["project"]["optional-dependencies"].items():
        for requirement in map(Requirement, specs):
            if requirement.name != "starlette":
                continue
            pins = list(requirement.specifier)
            assert len(pins) == 1 and pins[0].operator == "==", (extra, requirement)
            assert Version(pins[0].version) >= floor, (extra, requirement)
            found.add(extra)
    assert {"web", "mcp", "computer-use"} <= found
    dev = [req for req in map(Requirement, metadata["dependency-groups"]["dev"])
           if req.name == "starlette"]
    assert len(dev) == 1
    pins = list(dev[0].specifier)
    assert len(pins) == 1 and pins[0].operator == "==" and Version(pins[0].version) >= floor
    versions = [Version(row["version"]) for row in lock["package"] if row["name"] == "starlette"]
    assert versions and all(version >= floor for version in versions)


def test_exact_pinned_deps_exempt_from_exclude_newer():
    # `exclude-newer` filters a package the resolver cannot date, and the
    # global `exclude-newer = "14 days"` reaches every dependency that has no
    # entry here. An exact pin under that cutoff bricks whenever the index or
    # the uv HTTP cache omits its upload date, and because the lock covers
    # every extra, one such pin fails `uv lock` for all of them (PM relocks on
    # plugin enable/disable, `pm install`/`pm sync` and every source update).
    # An exact pin cannot float without a reviewed bump, so exempting it costs
    # no supply-chain safety — see the rationale block above
    # [tool.uv.exclude-newer-package] and #126194, where an unexempted
    # pilk==0.2.4 (on PyPI since 2023-05-26, published without upload dates)
    # did exactly that. `[dependency-groups]` is included because the lock
    # resolves every group, so an undated cutoff there fails the same way. The
    # brick-shape entries use `false` (no cutoff at all); a dated entry is the
    # narrower "admit exactly this release" intent and is accepted here. What
    # must not happen is an exact pin with no entry.
    metadata, exempt = _exclude_newer_exemptions()
    project = metadata["project"]
    specs = list(project.get("dependencies", []))
    for extra_specs in project.get("optional-dependencies", {}).values():
        specs.extend(extra_specs)
    for group_specs in metadata.get("dependency-groups", {}).values():
        specs.extend(group_specs)
    unexempted = sorted(
        f"{name} ({spec})"
        for name, spec in _exact_pins(specs)
        if _canonical(name) not in exempt
    )
    assert unexempted == [], (
        "exact-pinned dependencies must be listed in "
        "[tool.uv.exclude-newer-package]: " + ", ".join(unexempted)
    )


def test_build_system_requires_exempt_from_exclude_newer():
    # uv applies the cutoff to build-system.requires too, so an exact-pinned
    # build requirement is filtered on wheel-less platforms (Termux/Android)
    # where the pin has to build from an sdist in an isolated environment.
    metadata, exempt = _exclude_newer_exemptions()
    unexempted = sorted(
        f"{name} ({spec})"
        for name, spec in _exact_pins(metadata.get("build-system", {}).get("requires", []))
        if _canonical(name) not in exempt
    )
    assert unexempted == [], (
        "exact-pinned build requirements must be listed in "
        "[tool.uv.exclude-newer-package]: " + ", ".join(unexempted)
    )
