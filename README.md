# neural-networks

Learning notebooks, starting with a from-scratch reimplementation of
[micrograd](https://github.com/karpathy/micrograd) in `building-micrograd.ipynb`.

## Setup (do this once)

This project is managed with [`uv`](https://docs.astral.sh/uv/). From the repo root:

```bash
uv sync                  # create .venv and install the exact locked deps
brew install graphviz    # system 'dot' binary — required to render draw_dot()
```

`uv sync` builds `.venv/` with Python 3.12. The graphviz **Python package** is
locked, but rendering also needs the graphviz **system binary** (`dot`), which
Homebrew provides — without it `draw_dot(...)` raises `ExecutableNotFound`.

## Running the notebook in VSCode

1. Open `building-micrograd.ipynb`.
2. Kernel picker (top-right) → select the **`.venv/bin/python`** interpreter.
   Do **not** pick a global "Python 3.12.12" / "Python 3.14" — those don't have
   the project's packages.
3. Run the first cell (the environment check). It prints the interpreter path
   and every library version, and **asserts** you're on the `.venv` kernel with
   `dot` available. If that cell passes, the environment is correct.

## Gotchas this repo is pinned against

- **ipykernel is pinned `<7`.** ipykernel 7.x hangs VSCode's Jupyter extension on
  kernel connect (spinner that never resolves). Stay on the 6.x line.
- **numpy `<2` and torch `<2.3`** are pinned to a compatible pair.
- If a cell hangs with the kernel at **0% CPU**, it's not computing — the kernel
  is stuck/disconnected. Use **Restart Kernel**; don't wait it out.
- If imports fail "for no reason," the kernel is almost certainly the wrong
  interpreter. Re-run cell 1 — the assert will tell you.
