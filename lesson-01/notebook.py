import marimo

__generated_with = "0.21.1"
app = marimo.App(width="medium")


@app.cell
def _():
    import math
    import numpy as np
    import matplotlib.pyplot as plt
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    class ValueRenderer:
        @classmethod
        def render(cls, value):
            def _format_value_text(node):
                _label = f"{node.label}<br/>" if node.label else ""
                return f"{_label}data: {node.data}<br/>grad: {node.grad}".replace(
                    '"', '\\"'
                )

            def build_mermaid(node, visited):
                if node in visited:
                    return ""
                visited.add(node)

                value_id = f"value_{id(node)}"
                mermaid_str = f'{value_id}["{_format_value_text(node)}"]\n'

                target_id = value_id
                if node._op:
                    op_id = f"op_{id(node)}"
                    mermaid_str += f'{op_id}(("{node._op}"))\n'
                    mermaid_str += f"{op_id} --> {value_id}\n"
                    target_id = op_id

                for child in node._prev:
                    child_id = f"value_{id(child)}"
                    mermaid_str += f"{child_id} --> {target_id}\n"
                    mermaid_str += build_mermaid(child, visited)

                return mermaid_str

            visited = set()
            return mo.mermaid("graph TD\n" + build_mermaid(value, visited))

    return (ValueRenderer,)


@app.cell
def _(ValueRenderer):
    class Value:
        def __init__(self, data, _children=(), _op="", label=""):
            self.data = data
            self.grad = 0.0
            self._prev = set(_children)
            self._op = _op
            self.label = label

        def __repr__(self):
            return f"Value({self.data})"

        def __add__(self, other):
            out = Value(self.data + other.data, (self, other), "+")

            self._set_and_propagate_gradient(1.0)
            other._set_and_propagate_gradient(1.0)

            return out

        def __mul__(self, other):
            out = Value(self.data * other.data, (self, other), "*")

            self._set_and_propagate_gradient(other.data)
            other._set_and_propagate_gradient(self.data)

            return out

        def _set_and_propagate_gradient(self, grad):
            self.grad = grad

            for child in self._prev:
                child.grad *= self.grad

        def render(self):
            return ValueRenderer().render(self)


    a = Value(2.0, label="a")
    b = Value(-3.0, label="b")
    c = Value(10.0, label="c")
    e = a + b
    e.label = "e"
    d = e + c
    d.label = "d"
    f = Value(-2.0, label="f")
    L = d * f
    L.label = "L"
    L.render()
    return


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
