"""Vérifier les cellules dans leur ordre sans exécuter les notebooks."""

import json
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="telco-pyright-") as temporary:
        paths = []
        for notebook in sorted((ROOT / "notebooks").glob("*.ipynb")):
            document = json.loads(notebook.read_text())
            source = "\n\n".join(
                "".join(cell["source"]) for cell in document["cells"] if cell["cell_type"] == "code"
            )
            compile(source, str(notebook), "exec")
            target = Path(temporary) / f"{notebook.stem}.py"
            target.write_text(source)
            paths.append(str(target))
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "pyright",
                "--project",
                str(ROOT / "pyrightconfig.json"),
                "--pythonpath",
                sys.executable,
                str(ROOT / "settings.py"),
                *paths,
            ],
            cwd=ROOT,
            check=False,
        )
        return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
