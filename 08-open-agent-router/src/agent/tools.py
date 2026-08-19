"""Three real tools. No framework, no dependencies, and no mocks.

The compounding numbers in eval/compounding.py have to come from an episodic loop that
actually executes something, or p^n versus the measured end-to-end rate is a comparison
between two formulas. So these tools do real work: the calculator evaluates real
arithmetic, the file reader reads real bytes off disk, the search really scans lines.
What is simulated is only whether the *model* emitted a well-formed call -- that is the
one thing there is no model here to decide (src/fleet/simulator.py).

Safety. `calculate` does not call eval(). It walks Python's own AST and evaluates a
whitelist of node types, so an expression that is not arithmetic raises instead of
executing. eval() on model-emitted text in an agent loop is a remote code execution
primitive; the fact that this repo's "model" is a random number generator is not a
reason to write the pattern down.
"""

from __future__ import annotations

import ast
import operator
import pathlib
from dataclasses import dataclass

_BINOPS = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
    ast.Div: operator.truediv, ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod, ast.Pow: operator.pow,
}
_UNARYOPS = {ast.UAdd: operator.pos, ast.USub: operator.neg}


class ToolError(RuntimeError):
    """A tool refused the call. Distinct from a wrong answer -- the loop counts both."""


def calculate(expression: str) -> float:
    """Evaluate an arithmetic expression via a whitelisted AST walk.

    Exponents are capped: 9**9**9 is a valid arithmetic expression and a denial of
    service, and an agent tool that can be handed arbitrary model output has to assume
    the model is having a bad day.
    """
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as exc:
        raise ToolError(f"calculate: not a valid expression: {expression!r}") from exc

    def walk(node: ast.AST) -> float:
        if isinstance(node, ast.Expression):
            return walk(node.body)
        if isinstance(node, ast.Constant):
            if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
                raise ToolError(f"calculate: non-numeric constant {node.value!r}")
            return float(node.value)
        if isinstance(node, ast.BinOp) and type(node.op) in _BINOPS:
            left, right = walk(node.left), walk(node.right)
            if isinstance(node.op, ast.Pow) and (abs(right) > 32 or abs(left) > 1e6):
                raise ToolError("calculate: exponent out of the allowed range")
            if isinstance(node.op, (ast.Div, ast.FloorDiv, ast.Mod)) and right == 0:
                raise ToolError("calculate: division by zero")
            return _BINOPS[type(node.op)](left, right)
        if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARYOPS:
            return _UNARYOPS[type(node.op)](walk(node.operand))
        raise ToolError(f"calculate: disallowed syntax {type(node).__name__}")

    return float(walk(tree))


def read_file(path: str | pathlib.Path, root: str | pathlib.Path) -> str:
    """Read a file, refusing anything that escapes `root`.

    resolve() before the containment check, otherwise '../../etc/passwd' passes a
    string prefix test and fails the intent.
    """
    root = pathlib.Path(root).resolve()
    target = (root / path).resolve()
    if root not in target.parents and target != root:
        raise ToolError(f"read_file: {path!r} escapes the sandbox root {root}")
    if not target.is_file():
        raise ToolError(f"read_file: no such file: {path!r}")
    return target.read_text()


def search_text(needle: str, path: str | pathlib.Path,
                root: str | pathlib.Path) -> int:
    """Count the lines of a file containing `needle`. Case-sensitive, substring match."""
    text = read_file(path, root)
    return sum(1 for line in text.splitlines() if needle in line)


@dataclass(frozen=True)
class ToolCall:
    """A parsed tool call. The 'model' either emits this correctly or corrupts it."""

    tool: str
    args: dict


TOOLS = ("calculate", "read_file", "search_text")


def dispatch(call: ToolCall, root: str | pathlib.Path):
    """Execute a tool call. Raises ToolError on anything malformed."""
    if call.tool == "calculate":
        return calculate(call.args["expression"])
    if call.tool == "read_file":
        return read_file(call.args["path"], root)
    if call.tool == "search_text":
        return search_text(call.args["needle"], call.args["path"], root)
    raise ToolError(f"dispatch: unknown tool {call.tool!r}; known tools are {TOOLS}")
