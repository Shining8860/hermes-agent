"""Cron: interpreter and import path of the restart-safe external worker.

The worker is spawned as ``<payload-python> -m cron.scheduler``. Its entry module is
``cron.scheduler``, not ``hermes_cli.main``, so nothing bootstraps the gateway's checkout
onto its ``sys.path``; historically it imported ``cron`` only through the implicit ``-m``
cwd entry. That entry is gone under ``PYTHONSAFEPATH`` and useless when the venv's
editable install maps a moved/deleted checkout -- the worker then dies with
"No module named 'cron'" before its ownership ack (#112729, hypothesised cause).

The shared subprocess sanitizer strips Hermes-owned PYTHONPATH entries because user
children must not see our tree. This child IS Hermes, so the pin is applied *after* the
env is built, on the sanitized env. On a self-managed (shell-installer / PM) install the
sanitizer's drop of the runtime site-packages cannot stand this time: the worker inherits
this process's interpreter, which is PM's store Python and owns no third-party
dependencies, so the committed generation's site-packages is restored here too or the
child dies at its first dependency import (``No module named 'ruamel'``, #122222) before
it can publish its ownership acknowledgement.
"""

from __future__ import annotations

import logging
import os
import sys
import sysconfig
from pathlib import Path

logger = logging.getLogger(__name__)


def external_worker_python(repo_root: Path) -> Path:
    """Return the interpreter for the dependency environment active on ``sys.path``.

    A managed gateway starts on the bare store Python and ``hermes_bootstrap`` then adds
    the immutable project's site-packages in-process.  A direct ``-m cron.scheduler`` child
    has no such bootstrap, so inheriting ``sys.executable`` loses dependencies such as
    ``ruamel.yaml`` before it can acknowledge ownership.  Deriving the venv from the active
    site-packages also keeps source/legacy installs on their current interpreter and avoids
    consulting mutable profile state.  ``repo_root`` remains explicit for the shared worker
    API and future non-venv layouts.
    """
    del repo_root
    for entry in sys.path:
        path = Path(entry)
        if path.name != "site-packages":
            continue
        if path.parent.name.lower() == "lib":  # Windows: <venv>/Lib/site-packages
            candidate = path.parent.parent / "Scripts" / "python.exe"
        elif path.parent.parent.name == "lib":  # POSIX: <venv>/lib/pythonX.Y/site-packages
            candidate = path.parent.parent.parent / "bin" / "python"
        else:
            continue
        if candidate.is_file():
            return candidate
    return Path(sys.executable)


def _installed_purelib() -> Path | None:
    try:
        return Path(sysconfig.get_paths()["purelib"]).resolve()
    except (KeyError, OSError):
        return None


def _committed_dependency_site_packages(project_root: Path) -> Path | None:
    """The dependency ``site-packages`` PM committed for this install, or ``None``.

    Asked of PM's own committed selection -- the record ``activate_dependencies`` resolves
    at process boot -- rather than re-derived from this process's ``sys.path``: the record
    belongs to this install (``install_state_dir``, shared by every profile it serves). A
    runner that owns its dependencies (wheel / pipx / developer venv / Nix: no committed
    generation) has nothing to restore, so ``None`` means "pin the tree only" and nothing
    is invented.
    """
    try:
        from pm.environments import committed_venv, site_packages

        environment = committed_venv(project_root)
    except Exception as exc:
        # An unreadable record: pin the tree only. The cron worker's own boot re-reads it and
        # fails the dispatch with PM's error.
        logger.warning(
            "cron worker: could not read the committed dependency environment: %s", exc
        )
        return None
    if environment is None:
        return None
    selected = site_packages(environment)
    return selected if selected.is_dir() else None


def pin_hermes_tree_on_pythonpath(worker_env: dict, repo_root: Path) -> dict:
    """Prepend ``repo_root`` -- and, when this runner has one, the committed dependency
    generation's ``site-packages`` -- to the worker env's own PYTHONPATH (never
    ``os.environ``'s).

    Skipped when ``repo_root`` is the interpreter's ``purelib``: under a wheel / pipx /
    uv-tool install ``cron/`` lives in site-packages itself, which is already importable,
    and pinning it would move site-packages ahead of the stdlib on ``sys.path``.

    Order (checkout, generation, sanitizer-kept) mirrors ``activate_dependencies``' own
    ``sys.path``. For the cron worker, its boot (``cron/worker_bootstrap.py``) then re-selects
    and leases the committed generation before any third-party import, and exits the worker
    if it cannot.
    """
    root = str(repo_root)
    if _installed_purelib() == Path(root).resolve():
        return worker_env
    existing = [e for e in worker_env.get("PYTHONPATH", "").split(os.pathsep) if e]
    dependency = _committed_dependency_site_packages(Path(root))
    pinned = [root, *([str(dependency)] if dependency is not None else [])]
    worker_env["PYTHONPATH"] = os.pathsep.join(dict.fromkeys([*pinned, *existing]))
    return worker_env
