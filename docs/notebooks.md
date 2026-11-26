# Notebook conventions (branch `claude/gambit-full-design-notebooks`)

On this branch all Python code lives in Jupyter notebooks, so ideas can be written, tested
and annotated in one place (locally or in Google Colab). C/C++, CMake, YAML, SDF and ROS
launch files stay as normal files.

The one exception: `gambit_bringup/launch/gambit.launch.py` stays a `.py` file, because
`ros2 launch` can only load launch files from `.py`/`.xml`/`.yaml`.

## Layout

Each package keeps its notebooks in one flat folder:

```
gambit_<pkg>/notebooks/
    00_overview.ipynb        what the package does, deliverable checklist, run order
    <module>.ipynb           "library" notebook: definitions + quick self-checks
    run_<task>.ipynb         "run" notebook: talks to hardware, trains, or writes results
```

## Library vs run notebooks

* **Library notebooks** contain only definitions (functions, classes, constants) plus a
  final `## Self-check` section with fast asserts that need no hardware, GPU or data.
  Running one top to bottom must take seconds and must have no side effects.
* **Run notebooks** do the real work (camera capture, training, arm motion, evaluation).
  They pull in library notebooks and are never imported by anything else.

## Reusing code from another notebook

Use IPython's `%run`, with a path relative to the notebook's own folder:

```python
%run ./labels.ipynb                               # same package
%run ../../gambit_vision/notebooks/labels.ipynb   # another package
```

`%run` executes the whole notebook (including its self-check cells) and copies its
names into the current namespace. That is why library notebooks must stay fast and
side-effect free.

In Colab, first clone the repo and `%cd` into the notebook's folder so the relative
paths resolve:

```python
!git clone https://github.com/BrandonZabawa/Gambit-the-robot-chess-player.git
%cd Gambit-the-robot-chess-player/gambit_vision/notebooks
```

## Cell structure

1. Markdown title, one paragraph on what the notebook is for, its deliverable or
   milestone, and a **TODO** checklist.
2. Imports and `%run` dependencies.
3. Sections (`## ...`) that each have a markdown note followed by code. Unfinished code
   is marked `# TODO:` and raises `NotImplementedError`, so a skipped step fails loudly.
4. `## Self-check` with assert-based tests of whatever is already implemented.

## Executing every notebook (what CI does)

```bash
pip install nbclient ipykernel
tools/run_notebooks.sh                   # runs every library notebook (skips run_*.ipynb)
```
