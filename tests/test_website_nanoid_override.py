"""Pin/lock consistency for the website's nanoid override.

GHSA-2v37-7h3g-55p8 / CVE-2026-67213 affects nanoid <3.3.18 on the 3.x line.
The website is a standalone npm project, not a root workspace, so the root
override (package.json "nanoid@^3") does not reach it. This test keeps the
website override and every website lock resolution on the patched release, so
a future edit cannot silently re-pin the vulnerable version or regenerate the
lock against a different override.
"""

import json
from pathlib import Path

WEBSITE = Path(__file__).resolve().parents[1] / "website"
PATCHED_NANOID = "3.3.18"


def _load(name):
    return json.loads((WEBSITE / name).read_text(encoding="utf-8"))


def _version(value):
    return tuple(int(part) for part in value.split("."))


def _satisfies(version, spec):
    """Evaluate the range forms nanoid consumers actually declare.

    Unhandled forms fail loudly: the point is a contract, not a semver
    implementation. Extend this helper deliberately if a consumer changes.
    """
    spec = spec.strip()
    if spec.startswith("^"):
        low = _version(spec[1:])
        current = _version(version)
        return current >= low and current[0] == low[0]
    if spec.startswith(">="):
        return _version(version) >= _version(spec[2:])
    raise AssertionError(f"unhandled nanoid range {spec!r}")


def test_website_override_pins_the_patched_nanoid():
    override = _load("package.json")["overrides"]["nanoid"]
    assert override == PATCHED_NANOID, (
        f"website nanoid override is {override!r}; GHSA-2v37-7h3g-55p8 "
        f"requires {PATCHED_NANOID} on the 3.x line"
    )


def test_website_lock_resolves_nanoid_to_the_override():
    lock = _load("package-lock.json")
    assert lock["lockfileVersion"] == 3
    override = _load("package.json")["overrides"]["nanoid"]

    entries = {
        path: entry
        for path, entry in lock["packages"].items()
        if path.rsplit("node_modules/", 1)[-1] == "nanoid"
    }
    assert entries, "website lock contains no nanoid resolution"

    for path, entry in entries.items():
        assert entry["version"] == override, (path, entry["version"], override)
        assert entry["resolved"] == (
            f"https://registry.npmjs.org/nanoid/-/nanoid-{override}.tgz"
        ), path
        assert entry.get("integrity", "").startswith("sha512-"), path

    consumers = []
    for path, entry in lock["packages"].items():
        for field in ("dependencies", "optionalDependencies", "peerDependencies"):
            spec = entry.get(field, {}).get("nanoid")
            if spec is not None:
                consumers.append((path or "<root>", field, spec))
    assert consumers, "website lock declares no nanoid consumers"

    for path, field, spec in consumers:
        assert _satisfies(override, spec), (path, field, spec, override)
