import copy
from abc import ABC, abstractmethod
from functools import reduce
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Type, TypeVar

from . import mode
from .utils import join_not_null

__all__ = [
    "Graph",
    "Node",
    "Operation",
    "Rule",
    "Tensor",
    "Variable",
    "AttrMap",
    "GraphAttrKey",
]


class AttrMap:
    def __init__(
        self,
        attrs: Dict[str, Any] = None,
        ops: Dict[str, Dict[str, Any]] = None,
        tensors: Dict[str, Dict[str, Any]] = None,
    ):
        self.attrs = attrs or {}
        self.ops = ops or {}
        self.tensors = tensors or {}

    @property
    # @lru_cache()
    def nodes(self) -> Dict[str, Dict[str, Any]]:
        return {**self.ops, **self.tensors}


class GraphAttrKey:
    PREDICT = "predict"
    PREDICT_TOP5 = "predict_top5"
    PREDICT_TOP5_VALUE = "predict_top5_value"
    SEED = "seed"


class Graph:
    def __init__(self):
        self._current_node_id = 0
        self.ops: Set[Operation] = set()
        self.tensors: Set[Tensor] = set()
        self._op_by_id: Dict[int, Operation] = {}
        self._tensor_by_id: Dict[int, Tensor] = {}
        self._id_by_name: Dict[str, int] = {}
        self._inputs: List[int] = []
        self._outputs: List[int] = []
        self._attrs = {}

    @property
    def attrs(self) -> Dict[str, Any]:
        return self._attrs

    def attrs_to_map(self) -> AttrMap:
        return AttrMap(
            attrs=self.attrs,
            ops={op.name: op.attrs for op in self.ops},
            tensors={tensor.name: tensor.attrs for tensor in self.tensors},
        )

    def clone(self) -> "Graph":
        return copy.deepcopy(self)

    def with_attrs(self, map: AttrMap) -> "Graph":
        graph = copy.deepcopy(self)
        graph.load_attrs(map)
        return graph

    def load_attrs(self, map: AttrMap):
        self.attrs.update(map.attrs)
        for op in self.ops:
            op.attrs.update(map.ops[op.name])
        for tensor in self.tensors:
            tensor.attrs.update(map.tensors[tensor.name])

    def apply(self, func: Callable[["Graph"], "Graph"]):
        func(self)
        return self

    @property
    def variables(self) -> Dict[str, "Variable"]:
        return {
            variable.name: variable
            for op in self.ops
            for variable in op.variables
            if variable is not None
        }

    def layers(self) -> List[str]:
        layer_list = []

        def iterate_graph(node_id: int):
            node_name = self.node(node_id).name
            if node_name not in layer_list:
                layer_list.append(node_name)
                if node_id not in self.inputs:
                    node_inputs = self.node(node_id).inputs
                    if len(node_inputs) == 1:
                        iterate_graph(node_inputs[0])
                    else:
                        node_inputs = list(
                            filter(
                                lambda input_id: not (
                                    self.contains_tensor(input_id)
                                    and self.tensor(input_id).name.startswith("add")
                                ),
                                node_inputs,
                            )
                        )
                        # if len(node_inputs) != 1:
                        #     print(f"guess: choose first input in {node_name}")
                        iterate_graph(node_inputs[0])
            else:
                print(f"node {node_name} is already in list")

        for output_index, node_id in enumerate(self.outputs):
            iterate_graph(node_id)
        return list(reversed(layer_list))

    OP = TypeVar("OP", bound="Operation")

    def ops_in_layers(self, *op_types: Type[OP]) -> List[str]:
        if len(op_types) == 0:
            return list(
                filter(lambda layer: self.contains_op(self.id(layer)), self.layers())
            )
        else:
            op_types = tuple(op_types)
            return list(
                filter(
                    lambda layer: self.contains_op(self.id(layer))
                    and isinstance(self.op(self.id(layer)), op_types),
                    self.layers(),
                )
            )

    def gc(self):
        reachable_nodes: Set[int] = set()

        def iterate_graph(node_id: int):
            if node_id not in reachable_nodes:
                reachable_nodes.add(node_id)
                if node_id not in self.inputs:
                    for input_id in self.node(node_id).inputs:
                        iterate_graph(input_id)

        for output_index, node_id in enumerate(self.outputs):
            iterate_graph(node_id)
        self._op_by_id = {
            node_id: self.op(node_id)
            for node_id in reachable_nodes
            if self.contains_op(node_id)
        }
        self._tensor_by_id = {
            node_id: self.tensor(node_id)
            for node_id in reachable_nodes
            if self.contains_tensor(node_id)
        }
        self._id_by_name = {
            self.node(node_id).name: node_id for node_id in reachable_nodes
        }
        self.ops = set(self._op_by_id.values())
        self.tensors = set(self._tensor_by_id.values())
        for node in self.nodes:
            for input_id in list(node.inputs):
                if input_id not in self:
                    node.inputs.remove(input_id)
            for output_id in list(node.outputs):
                if output_id not in self:
                    node.outputs.remove(output_id)

    def rename(self, node_id: int, new_name: str):
        node = self.node(node_id)
        del self._id_by_name[node.name]
        self._id_by_name[new_name] = node_id
        node._name = new_name

    def print(self):
        reachable_nodes: Set[int] = set()

        def iterate_graph(node_id: int, level: int):
            if node_id not in reachable_nodes:
                reachable_nodes.add(node_id)
                print("  " * level + str(self.node(node_id)))
                if node_id not in self.inputs:
                    for input_id in self.node(node_id).inputs:
                        iterate_graph(input_id, level + 1)

        for output_index, node_id in enumerate(self.outputs):
            iterate_graph(node_id, 0)

    @property
    def nodes(self) -> List["Node"]:
        return self.ops.union(self.tensors)

    @property
    def inputs(self) -> List[int]:
        return self._inputs

    @property
    def outputs(self) -> List[int]:
        return self._outputs

    def contains_name(self, name: str) -> bool:
        return name in self._id_by_name

    def __contains__(self, node_id: int) -> bool:
        return node_id in self._op_by_id or node_id in self._tensor_by_id

    def contains_op(self, op_id: int) -> bool:
        return op_id in self._op_by_id

    def contains_tensor(self, tensor_id: int) -> bool:
        return tensor_id in self._tensor_by_id

    def id(self, name: str) -> int:
        return self._id_by_name[name]

    def node(self, node_id: int) -> "Node":
        if self.contains_op(node_id):
            return self._op_by_id[node_id]
        else:
            return self._tensor_by_id[node_id]

    def op(self, op_id: int) -> "Operation":
        return self._op_by_id[op_id]

    def tensor(self, tensor_id: int) -> "Tensor":
        return self._tensor_by_id[tensor_id]

    def _next_node_id(self) -> int:
        next_id = self._current_node_id
        self._current_node_id += 1
        return next_id

    def add_op(self, op: "Operation") -> int:
        next_id = self._next_node_id()
        if op in self.ops:
            raise RuntimeError(f"op {op.name} has been added")
        else:
            self._op_by_id[next_id] = op
            self._id_by_name[op.name] = next_id
            self.ops.add(op)
            return next_id

    def add_tensor(self, tensor: "Tensor") -> int:
        next_id = self._next_node_id()
        if tensor in self.tensors:
            raise RuntimeError(f"tensor {tensor.name} has been added")
        else:
            self._tensor_by_id[next_id] = tensor
            self._id_by_name[tensor.name] = next_id
            self.tensors.add(tensor)
            return next_id

    def add_node(self, node: "Node") -> int:
        if isinstance(node, Operation):
            return self.add_op(node)
        elif isinstance(node, Tensor):
            return self.add_tensor(node)
        else:
            raise RuntimeError(f"node {node.name} is neither Op nor Tensor")

    def assert_contains_tensor(self, *tensor_ids: int):
        for tensor_id in tensor_ids:
            if not self.contains_tensor(tensor_id):
                raise RuntimeError(f"tensor with id {tensor_id} is not in this graph")

    def assert_contains_op(self, *op_ids: int):
        for op_id in op_ids:
            if not self.contains_op(op_id):
                raise RuntimeError(f"op with id {op_id} is not in this graph")

    def add_input(self, input_id: int):
        self.assert_contains_tensor(input_id)
        if input_id not in self.inputs:
            self.inputs.append(input_id)
        else:
            raise RuntimeError(
                f"input {self.tensor(input_id).name} with id {input_id} has been added"
            )

    def add_output(self, output_id: int):
        self.assert_contains_tensor(output_id)
        if output_id not in self.outputs:
            self.outputs.append(output_id)
        else:
            raise RuntimeError(
                f"output {self.tensor(output_id).name} with id {output_id} has been added"
            )

    def rewrite(self, *rules: "Rule") -> "Graph":
        def apply_rule(result: Tuple["Graph", bool], rule: "Rule"):
            graph, already_changed = result
            new_graph, changed = rule(graph)
            return new_graph, already_changed or changed

        graph = self
        changed = True
        while changed:
            graph, changed = reduce(apply_rule, rules, (graph, False))
        graph.gc()
        # if mode.is_check():
        #     graph.print()
        return graph


class Node(ABC):
    def __init__(self, graph: Graph, name: str = ""):
        self._graph = graph
        self._name = name
        self._inputs = []
        self._outputs = []
        self._id = graph.add_node(self)
        self._attrs = {}

    @property
    def graph(self) -> Graph:
        return self._graph

    @property
    def name(self) -> str:
        return self._name

    @property
    def id(self) -> int:
        return self._id

    @property
    def inputs(self) -> List[int]:
        return self._inputs

    @property
    def outputs(self) -> List[int]:
        return self._outputs

    @property
    def input_nodes(self) -> List["Node"]:
        return [self.graph.node(node_id) for node_id in self.inputs]

    @property
    def output_nodes(self) -> List["Node"]:
        return [self.graph.node(node_id) for node_id in self.outputs]

    @property
    def attrs(self) -> Dict[str, Any]:
        return self._attrs

    @property
    def is_tensor(self) -> bool:
        return False

    @property
    def is_op(self) -> bool:
        return False

    @abstractmethod
    def add_input(self, input_id: int, ignore_if_added: bool = False):
        ...

    @abstractmethod
    def add_output(self, output_id: int, ignore_if_added: bool = False):
        ...

    @abstractmethod
    def add_edge(self, target_node_id: int):
        ...

    def remove_input(self, input_id: int) -> bool:
        if input_id in self.inputs:
            self.inputs.remove(input_id)
            return True
        else:
            return False

    def remove_output(self, output_id: int) -> bool:
        if output_id in self.outputs:
            self.outputs.remove(output_id)
            return True
        else:
            return False


class Variable:
    def __init__(self, name: str, value: Any = None):
        self._name = name
        self.value = value

    @property
    def name(self):
        return self._name

    def __str__(self):
        content = join_not_null(
            [
                '"' + self.name + '"',
                None if self.value is None else f"value={self.value}",
            ]
        )
        return f"Variable({content})"


class Tensor(Node):
    def __init__(self, graph: Graph, name: str = "", shape=None, dtype=None):
        super().__init__(graph, name)
        self.value = None
        self.shape = shape
        self.dtype = dtype

    @property
    def op(self) -> Optional["Operation"]:
        return None if self.op_id is None else self.graph.op(self.op_id)

    @property
    def op_id(self) -> Optional[int]:
        return None if len(self.inputs) == 0 else self.inputs[0]

    @op_id.setter
    def op_id(self, op: int):
        self.add_input(op)

    def add_input(self, input_id: int, ignore_if_added: bool = False):
        self.graph.assert_contains_op(input_id)
        if len(self.inputs) == 0:
            self.inputs.append(input_id)
        elif not ignore_if_added:
            raise RuntimeError(f"tensor only allow one input op")

    def add_output(self, output_id: int, ignore_if_added: bool = False):
        self.graph.assert_contains_op(output_id)
        if output_id not in self.outputs:
            self.outputs.append(output_id)
        elif not ignore_if_added:
            raise RuntimeError(
                f"output {self.graph.op(output_id).name} with id {output_id} has been added"
            )

    def add_edge(self, target_node_id: int):
        self.add_output(target_node_id)
        self.graph.op(target_node_id).add_input(self.id)

    @property
    def is_tensor(self) -> bool:
        return True

    def __str__(self) -> str:
        content = join_not_null(
            [
                '"' + self.name + '"',
                f"id={self.id}",
                None if self.value is None else f"value={self.value}",
                f"input={self.inputs}",
                f"outputs={self.outputs}",
            ]
        )
        return f"Tensor({content})"


class Operation(Node):
    def __init__(self, graph: Graph, name: str = ""):
        super().__init__(graph, name)
        self._variables = []

    @property
    def variables(self) -> List[Variable]:
        return self._variables

    def add_input(self, input_id: int, ignore_if_added: bool = False):
        self.graph.assert_contains_tensor(input_id)
        if input_id not in self.inputs:
            self.inputs.append(input_id)
        elif not ignore_if_added:
            raise RuntimeError(
                f"input {self.graph.tensor(input_id).name} with id {input_id} has been added"
            )

    def add_output(self, output_id: int, ignore_if_added: bool = False):
        self.graph.assert_contains_tensor(output_id)
        if output_id not in self.outputs:
            self.outputs.append(output_id)
        elif not ignore_if_added:
            raise RuntimeError(
                f"output {self.graph.tensor(output_id).name} with id {output_id} has been added"
            )

    def add_edge(self, target_node_id: int):
        self.add_output(target_node_id)
        self.graph.tensor(target_node_id).add_input(self.id)

    @property
    def is_op(self) -> bool:
        return True

    def __str__(self) -> str:
        content = join_not_null(
            [
                '"' + self.name + '"',
                f"id={self.id}",
                f"input={self.inputs}",
                f"outputs={self.outputs}",
            ]
        )
        return f"{type(self).__name__}({content})"


class Rule(ABC):
    @abstractmethod
    def action(self, node: Node) -> Any:
        ...

    def __call__(self, graph: Graph) -> Tuple[Graph, bool]:
        def replace_inputs_outputs(old_node_id: int, new_node_id: int):
            if old_node_id in graph.inputs:
                graph.inputs[graph.inputs.index(old_node_id)] = new_node_id
            if old_node_id in graph.outputs:
                graph.outputs[graph.outputs.index(old_node_id)] = new_node_id

        def apply_action(node_id: int, action: Any) -> int:
            def update_node(node_id: int, action: Any):
                if isinstance(action, int):
                    new_node_id = action
                    if node_id != new_node_id:
                        old_node = graph.node(node_id)
                        new_node = graph.node(new_node_id)
                        for input_node_id in old_node.inputs:
                            input_node_outputs = graph.node(input_node_id).outputs
                            input_node_outputs[
                                input_node_outputs.index(node_id)
                            ] = new_node_id
                            new_node.add_input(input_node_id, ignore_if_added=True)
                        old_node.inputs.clear()
                        for output_node_id in old_node.outputs:
                            output_node_inputs = graph.node(output_node_id).inputs
                            output_node_inputs[
                                output_node_inputs.index(node_id)
                            ] = new_node_id
                            new_node.add_output(output_node_id, ignore_if_added=True)
                        old_node.outputs.clear()
                        replace_inputs_outputs(node_id, new_node_id)
                elif isinstance(action, dict):

                    def normalized_list(value: Any, old_list: List[int]) -> List[int]:
                        if isinstance(value, list) and len(value) == len(old_list):
                            return value
                        elif isinstance(value, int):
                            return [value] * len(old_list)
                        elif isinstance(value, dict):
                            new_list = list(old_list)
                            for index, new_value in value.items():
                                new_list[index] = new_value
                            return new_list
                        else:
                            raise RuntimeError(
                                f"cannot normalize {value} into int list with size {len(old_list)}"
                            )

                    old_node = graph.node(node_id)
                    if "input" in action:
                        new_inputs = normalized_list(action["input"], old_node.inputs)
                        for input_node_id, new_input_node_id in zip(
                            old_node.inputs.copy(), new_inputs
                        ):
                            if input_node_id != new_input_node_id:
                                input_node_outputs = graph.node(input_node_id).outputs
                                input_node_outputs[
                                    input_node_outputs.index(node_id)
                                ] = new_input_node_id
                                graph.node(new_input_node_id).add_input(
                                    input_node_id, ignore_if_added=True
                                )
                                old_node.inputs.remove(input_node_id)
                    if "output" in action:
                        new_outputs = normalized_list(
                            action["output"], old_node.outputs
                        )
                        for output_node_id, new_output_node_id in zip(
                            old_node.outputs.copy(), new_outputs
                        ):
                            if output_node_id != new_output_node_id:
                                output_node_inputs = graph.node(output_node_id).inputs
                                output_node_inputs[
                                    output_node_inputs.index(node_id)
                                ] = new_output_node_id
                                graph.node(new_output_node_id).add_output(
                                    output_node_id, ignore_if_added=True
                                )
                                old_node.outputs.remove(output_node_id)
                    if "to" in action:
                        replace_inputs_outputs(node_id, action["to"])
                else:
                    raise RuntimeError(f"{action} is ill-formed")

            if action is None:
                return node_id
            elif isinstance(action, int):
                update_node(node_id, action)
                next_node_id = action
                if mode.is_check():
                    print({node_id: next_node_id})
                return next_node_id
            elif isinstance(action, dict):
                if mode.is_check():
                    print(action)
                for old_node_id, node_action in action.items():
                    if old_node_id != "next":
                        update_node(old_node_id, node_action)
                if "next" in action:
                    next_node_id = action["next"]
                else:
                    next_node_id = action[node_id]["to"]
                return next_node_id
            else:
                raise RuntimeError(f"{action} is ill-formed")

        visited_nodes: Set[int] = set()

        def iterate_graph(node_id: int):
            if node_id not in visited_nodes:
                current_node_id = node_id
                while True:
                    action = self.action(graph.node(current_node_id))
                    if action is None:
                        break
                    else:
                        current_node_id = apply_action(current_node_id, action)
                        changed[0] = True
                visited_nodes.add(current_node_id)
                if node_id not in graph.inputs:
                    for input_id in graph.node(current_node_id).inputs:
                        iterate_graph(input_id)

        changed = [False]
        for output_index, node_id in enumerate(graph.outputs):
            iterate_graph(node_id)
        return graph, changed[0]
