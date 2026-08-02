import marimo

__generated_with = "0.23.16"
app = marimo.App(width="medium")


@app.cell
def _():
    from pathlib import Path

    return (Path,)


@app.cell
def _(Path):
    text = Path("./shakespeare.txt").read_text(encoding="utf-8")
    return


if __name__ == "__main__":
    app.run()
