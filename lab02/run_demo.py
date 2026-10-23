"""Выполняет lab02.ipynb с настоящим перезапуском ядра.

Ячейки до markdown-ячейки с тегом "restart" выполняются в одном ядре, начиная с неё — в новом.
Результат с выводами записывается обратно в ноутбук.

    python run_demo.py lab02.ipynb
"""

import sys
from pathlib import Path

import nbformat
from nbclient import NotebookClient


def main(path: Path):
    nb = nbformat.read(path, as_version=4)
    split = next(i for i, c in enumerate(nb.cells) if "restart" in c.metadata.get("tags", []))
    for part in (range(0, split), range(split, len(nb.cells))):
        client = NotebookClient(nb, timeout=1800, kernel_name="python3",
                                resources={"metadata": {"path": str(path.parent)}})
        with client.setup_kernel():  # новое ядро на каждую часть — это и есть перезапуск
            for i in part:
                client.execute_cell(nb.cells[i], i)
                nbformat.write(nb, path)  # сохраняем по ходу, чтобы при ошибке было видно, где
        print(f"ячейки {part.start}–{part.stop - 1}: ядро завершено", flush=True)
    nbformat.write(nb, path)


if __name__ == "__main__":
    main(Path(sys.argv[1]).resolve())
