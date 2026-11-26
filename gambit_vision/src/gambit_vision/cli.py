"""Single command-line entry point: `gambit-vision <command>` (run from gambit_vision/).

Command bodies stay thin: parse args, load config, call a function in the stage module.
Most stage functions are still TODO skeletons; see docs/M3_TODO.md, M4_TODO.md, M5_TODO.md.
"""

from pathlib import Path
from typing import Optional

import typer

app = typer.Typer(no_args_is_help=True, help="Gambit Vision pipeline commands.")

DEFAULT_CFG = Path("configs/train.yaml")


def _cfg(path: Path):
    from gambit_vision.config import load_config

    return load_config(path)


@app.command()
def version() -> None:
    """Print the package version."""
    from gambit_vision import __version__

    typer.echo(__version__)


# ---------------------------------------------------------------- M3: training
@app.command("list-sessions")
def list_sessions(config: Path = DEFAULT_CFG) -> None:
    """List capture sessions under data.crops_root with their crop counts."""
    from gambit_vision.dataset import list_sessions as _list, load_manifest

    root = _cfg(config).data.crops_root
    for s in _list(root):
        typer.echo(f"{s}\t{len(load_manifest(root, [s]))} crops")


@app.command("train-baseline")
def train_baseline(config: Path = DEFAULT_CFG,
                   out: Path = Path("models/baseline/baseline.pkl")) -> None:
    """Fit the color/edge-feature baseline (occupancy + piece color)."""
    from gambit_vision.baseline import train_baseline as _train

    _train(_cfg(config), out_path=out)
    typer.echo(f"saved {out}")


@app.command("train-cnn")
def train_cnn(config: Path = DEFAULT_CFG, run_name: Optional[str] = None) -> None:
    """Fine-tune ResNet-18 on the 13 square classes."""
    from gambit_vision.train import train_cnn as _train

    typer.echo(f"best checkpoint: {_train(_cfg(config), run_name=run_name)}")


# ------------------------------------------------------- M4: assembly + eval
@app.command()
def evaluate(checkpoint: Path, config: Path = DEFAULT_CFG,
             run_name: Optional[str] = None) -> None:
    """Evaluate a CNN checkpoint on the test sessions; writes results/<run_name>/.

    The baseline is not 13-class (occupancy/color only), so it is evaluated
    separately; see docs/M4_TODO.md.
    """
    from gambit_vision.evaluate import evaluate_model
    from gambit_vision.train import CNNClassifier

    cfg = _cfg(config)
    if run_name:
        cfg.eval.run_name = run_name
    metrics = evaluate_model(cfg, CNNClassifier.from_checkpoint(checkpoint, cfg))
    typer.echo(metrics)


@app.command()
def assemble(probs: Path, threshold: float = 0.6) -> None:
    """Debug: turn a saved (64, 13) .npy probability array into a FEN."""
    import numpy as np

    from gambit_vision.board_assembly import assemble as _assemble

    pred = _assemble(np.load(probs), threshold=threshold)
    typer.echo(f"{pred.fen}  legal={pred.legal}  low_conf={pred.low_confidence}  issues={pred.issues}")


@app.command("chesscog-compare")
def chesscog_compare(ours: Path, chesscog: Path, out: Path) -> None:
    """Join our per-board predictions with chesscog's into chesscog_vs_ours.csv."""
    from gambit_vision.chesscog_compare import compare

    compare(ours, chesscog, out)
    typer.echo(f"wrote {out}")


# ------------------------------------------------------- M5: move detection
@app.command("eval-moves")
def eval_moves(pgn: Path, frames_dir: Path) -> None:
    """Replay a recorded game and report move-detection accuracy (TODO in move_detection)."""
    from gambit_vision.move_detection import evaluate_on_game

    typer.echo(evaluate_on_game(pgn, frames_dir))


if __name__ == "__main__":
    app()
