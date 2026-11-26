"""Single command-line entry point: `gambit-vision <command>`.

Each milestone registers its commands here, e.g.
    gambit-vision train-baseline --config configs/train.yaml
    gambit-vision train-cnn --config configs/train.yaml
    gambit-vision evaluate --config configs/eval.yaml
    gambit-vision detect-moves --video ...
Keep command bodies thin: parse args, load config, call a function in the stage module.
"""

import typer

app = typer.Typer(no_args_is_help=True, help="Gambit Vision pipeline commands.")


@app.command()
def version() -> None:
    """Print the package version."""
    from gambit_vision import __version__

    typer.echo(__version__)


if __name__ == "__main__":
    app()
