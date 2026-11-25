"""Execute all source notebooks with this Python environment, preserving evidence.

Run from any directory: python scripts/execute_notebooks.py. Results are saved
under outputs/notebooks/executed, with a JSON execution status record. No result
cells are fabricated, and failure exits nonzero after recording the traceback.
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from copy import deepcopy
from datetime import UTC, datetime
from pathlib import Path

import nbformat
from jupyter_client import KernelManager
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]


class ProjectKernelManager(KernelManager):
    """Use the running project interpreter even if another python is on PATH."""

    def format_kernel_cmd(self, extra_arguments=None):
        command = super().format_kernel_cmd(extra_arguments)
        command[0] = sys.executable
        return command


def portable_results(notebook):
    """Keep observed result displays/figures while excluding local console logs.

    Full executed copies, including logs and local paths, remain in ignored
    outputs. These notebooks use display() for all computed scientific results.
    """
    portable = deepcopy(notebook)
    for cell in portable.cells:
        if cell.cell_type == "code":
            cell.outputs = [output for output in cell.outputs if output.output_type != "stream"]
    return portable


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/notebooks/executed")
    parser.add_argument("--timeout", type=int, default=300, help="Per-cell timeout in seconds")
    parser.add_argument("--in-place", action="store_true", help="Also replace source notebooks with observed executed copies")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    statuses = []
    failed = False
    for source in sorted((ROOT / "notebooks").glob("0*.ipynb")):
        started = datetime.now(UTC)
        notebook = nbformat.read(source, as_version=4)
        status = {"notebook": source.name, "start_utc": started.isoformat(), "python": sys.executable}
        try:
            NotebookClient(notebook, timeout=args.timeout, startup_timeout=30, kernel_name="python3", kernel_manager_class=ProjectKernelManager,
                           resources={"metadata": {"path": str(ROOT)}}).execute()
            status["status"] = "passed"
            status["executed_code_cells"] = sum(cell.cell_type == "code" and cell.execution_count is not None for cell in notebook.cells)
            if args.in_place:
                nbformat.write(portable_results(notebook), source)
        except Exception:  # noqa: BLE001 -- preserve traceback for every unexpected notebook failure.
            status["status"] = "failed"
            status["traceback"] = traceback.format_exc()
            failed = True
        finally:
            nbformat.write(notebook, args.output / source.name)
            status["end_utc"] = datetime.now(UTC).isoformat()
            statuses.append(status)
            print(json.dumps({key: value for key, value in status.items() if key != "traceback"}), flush=True)
            (args.output / "execution_status.json").write_text(json.dumps(statuses, indent=2), encoding="utf-8")
        if failed:
            print(status["traceback"], file=sys.stderr)
            break
    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())
