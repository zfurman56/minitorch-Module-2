from dataclasses import dataclass
from typing import Any, Iterable, List, Tuple

from typing_extensions import Protocol

# ## Task 1.1
# Central Difference calculation


def central_difference(f: Any, *vals: Any, arg: int = 0, epsilon: float = 1e-6) -> Any:
    r"""
    Computes an approximation to the derivative of `f` with respect to one arg.

    See :doc:`derivative` or https://en.wikipedia.org/wiki/Finite_difference for more details.

    Args:
        f : arbitrary function from n-scalar args to one value
        *vals : n-float values $x_0 \ldots x_{n-1}$
        arg : the number $i$ of the arg to compute the derivative
        epsilon : a small constant

    Returns:
        An approximation of $f'_i(x_0, \ldots, x_{n-1})$
    """
    vals = list(vals)
    f_value = f(*vals)

    vals[arg] += epsilon
    delta_f_value = f(*vals)

    return (delta_f_value - f_value) / epsilon


variable_count = 1


class Variable(Protocol):
    def accumulate_derivative(self, x: Any) -> None:
        pass

    @property
    def unique_id(self) -> int:
        pass

    def is_leaf(self) -> bool:
        pass

    def is_constant(self) -> bool:
        pass

    @property
    def parents(self) -> Iterable["Variable"]:
        pass

    def chain_rule(self, d_output: Any) -> Iterable[Tuple["Variable", Any]]:
        pass


def topological_sort(variable: Variable) -> Iterable[Variable]:
    """
    Computes the topological order of the computation graph.

    Args:
        variable: The right-most variable

    Returns:
        Non-constant Variables in topological order starting from the right.
    """
    sorted_vars = []
    temp_marks = set()
    perm_marks = set()

    def visit(var):
        if var.unique_id in perm_marks:
            return
        if var.unique_id in temp_marks:
            raise Exception("Topological sort failed - graph has cycle!")

        temp_marks.add(var.unique_id)

        for parent in var.parents:
            visit(parent)

        temp_marks.remove(var.unique_id)
        perm_marks.add(var.unique_id)
        sorted_vars.append(var)

    visit(variable)
    sorted_vars.reverse()

    return sorted_vars


def backpropagate(variable: Variable, deriv: Any) -> None:
    """
    Runs backpropagation on the computation graph in order to
    compute derivatives for the leave nodes.

    Args:
        variable: The right-most variable
        deriv  : Its derivative that we want to propagate backward to the leaves.

    No return. Should write to its results to the derivative values of each leaf through `accumulate_derivative`.
    """

    derivs = {variable.unique_id: deriv}
    variables = topological_sort(variable)

    for var in variables:
        if var.is_leaf():
            var.accumulate_derivative(derivs[var.unique_id])
        else:
            for v, d in var.chain_rule(derivs[var.unique_id]):
                if v.unique_id not in derivs.keys():
                    derivs[v.unique_id] = d
                else:
                    derivs[v.unique_id] += d


@dataclass
class Context:
    """
    Context class is used by `Function` to store information during the forward pass.
    """

    no_grad: bool = False
    saved_values: Tuple[Any, ...] = ()

    def save_for_backward(self, *values: Any) -> None:
        "Store the given `values` if they need to be used during backpropagation."
        if self.no_grad:
            return
        self.saved_values = values

    @property
    def saved_tensors(self) -> Tuple[Any, ...]:
        return self.saved_values
