import marimo

__generated_with = "0.21.1"
app = marimo.App(width="medium")


@app.cell
def _():
    import math
    import numpy as np
    import matplotlib.pyplot as plt
    import marimo as mo
    import pytest
    from graphlib import TopologicalSorter
    import torch

    return TopologicalSorter, math, mo, pytest, torch


@app.cell(hide_code=True)
def _(mo):
    class ValueRenderer:
        @classmethod
        def render(cls, value):
            def _format_value_text(node):
                _label = f"{node.label}<br/>" if node.label else ""
                return f"{_label}data: {node.data:.2f}<br/>grad: {node.grad:.2f}".replace(
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
def _(Float, TopologicalSorter, ValueRenderer, math):
    class Value:
        def __init__(self, data, _children=(), _op="", label=""):
            self._backward = lambda: None
            self.data = data
            self.grad = 0.0
            self._prev = set(_children)
            self._op = _op
            self.label = label

        def __repr__(self):
            return f"Value({self.data:.2f}, label={self.label})"

        def __add__(self, other):
            other = other if isinstance(other, Value) else Value(other)
            out = Value(self.data + other.data, (self, other), "+")

            def _backward():
                self.grad += out.grad * 1.0
                other.grad += out.grad * 1.0

            out._backward = _backward
            return out

        def __neg__(self):
            return -1 * self

        def __sub__(self, other):
            return self + (-other)

        def __mul__(self, other):
            other = other if isinstance(other, Value) else Value(other)
            out = Value(self.data * other.data, (self, other), "*")

            def _backward():
                self.grad += out.grad * other.data
                other.grad += out.grad * self.data

            out._backward = _backward
            return out

        def __pow__(self, other):
            assert isinstance(other, int) or isinstance(other, Float)
            out = Value(self.data**other, (self,), f"^{other}")

            def _backward():
                self.grad += out.grad * other * (self.data ** (other - 1))

            out._backward = _backward
            return out

        def __rmul__(self, other):
            return self * other

        def __radd__(self, other):
            return self + other

        def __truediv__(self, other):
            return self * (other**-1)

        def exp(self):
            e_x = math.exp(self.data)
            out = Value(e_x, (self,), "e^x")

            def _backward():
                self.grad += out.grad * e_x

            out._backward = _backward

            return out

        def tanh(self):
            x = self.data
            e_2x = math.exp(2 * x)
            t = (e_2x - 1) / (e_2x + 1)
            out = Value(t, (self,), "tanh")

            def _backward():
                self.grad += out.grad * (1 - t**2)

            out._backward = _backward

            return out

        def backward(self):
            self.grad = 1

            for node in self._sorted_node_list():
                node._backward()

        def render(self):
            return ValueRenderer().render(self)

        def _sorted_node_list(self):
            graph = {}

            def build_graph(node):
                if node not in graph:
                    graph[node] = node._prev
                    for child in node._prev:
                        build_graph(child)

            build_graph(self)

            return reversed(list(TopologicalSorter(graph).static_order()))


    # inputs x1,x2
    x1 = Value(2.0, label="x1")
    x2 = Value(0.0, label="x2")
    # weights w1,w2
    w1 = Value(-3.0, label="w1")
    w2 = Value(1.0, label="w2")
    # bias of the neuron
    b = Value(6.8813735878195432, label="b")
    # x1*w1 + x2*w2 + b
    x1w1 = x1 * w1
    x1w1.label = "x1*w1"
    x2w2 = x2 * w2
    x2w2.label = "x2*w2"
    x1w1x2w2 = x1w1 + x2w2
    x1w1x2w2.label = "x1*w1 + x2*w2"
    n = x1w1x2w2 + b
    n.label = "n"
    o = n.tanh()
    o.label = "o"

    o.backward()
    o.render()
    return (Value,)


@app.cell
def _(Value, math, pytest):
    class TestValue:
        @staticmethod
        def test_basic_add_grad():
            a = Value(2.0)
            b = Value(3.0)
            c = a + b
            c.backward()
            assert b.grad == 1.0
            assert a.grad == 1.0

        @staticmethod
        def test_repeated_add_2_grad():
            a = Value(2.0)
            b = a + a
            b.backward()
            assert a.grad == 2.0

        @staticmethod
        def test_repeated_add_3_grad():
            a = Value(2.0)
            b = a + a + a
            b.backward()
            print(list(b._sorted_node_list()))
            assert a.grad == 3.0

        @staticmethod
        def test_repeated_mult_2_grad():
            a = Value(2.0)
            b = a * a
            b.backward()
            # db/da = 2 * a => 2.0
            assert a.grad == 4.0

        @staticmethod
        def test_repeated_mult_3_grad():
            a = Value(2.0)
            b = a * a * a
            b.backward()
            # db/da = 3 * a ^ 2 => 12.0
            assert a.grad == 12.0

        @staticmethod
        def test_tanh_forward():
            a = Value(2.0)
            b = a.tanh()
            expected = math.tanh(2.0)
            assert b.data == pytest.approx(expected)

        @staticmethod
        def test_tanh_grad():
            a = Value(2.0)
            b = a.tanh()
            b.backward()
            expected_grad = 1 - math.tanh(2.0) ** 2
            assert a.grad == pytest.approx(expected_grad)

    return


@app.cell
def _(torch):
    def use_torch():
        x1 = torch.Tensor([2.0]).double()
        x2 = torch.Tensor([0.0]).double()
        w1 = torch.Tensor([-3.0]).double()
        w2 = torch.Tensor([1.0]).double()
        b = torch.Tensor([6.8813735878195432]).double()
    
        for leaf_node in [x1, x2, w1, w2, b]:
            leaf_node.requires_grad = True
    
        return torch.tanh(x1*w1 + x2*w2 + b)

    use_torch()
    return


if __name__ == "__main__":
    app.run()
