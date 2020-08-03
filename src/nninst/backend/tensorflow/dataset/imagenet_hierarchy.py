from typing import Dict, Iterable, List, Set

from nninst.utils.fs import IOAction, abspath


class Node:
    def __init__(self, tree: "Tree", label: str, parent: "Node" = None):
        self.tree = tree
        if parent is not None:
            self.parents = [parent]
        else:
            self.parents: List["Node"] = []
        self.children = []
        self.label = label


class Tree:
    def __init__(self):
        self.label_to_node: Dict[str, Node] = {}
        self.imagenet_labels: List[str] = []
        self.roots: List[Node] = []

    def label(self, class_id: int) -> str:
        self.imagenet_labels[class_id]

    def nodes(self) -> Iterable[Node]:
        return self.label_to_node.values()

    def node(self, label: str) -> Node:
        return self.label_to_node[label]

    def has_node(self, label: str) -> bool:
        return label in self.label_to_node

    def add_edge(self, parent: str, child: str):
        if not self.has_node(parent):
            self.label_to_node[parent] = Node(self, label=parent)
        if not self.has_node(child):
            self.label_to_node[child] = Node(
                self, label=child, parent=self.node(parent)
            )
        else:
            assert self.node(parent) not in self.node(child).parents
            self.node(child).parents.append(self.node(parent))
        assert self.node(child) not in self.node(parent).children
        self.node(parent).children.append(self.node(child))

    def print(self):
        def iterate_graph(node: Node, level: int):
            if node not in reachable_nodes:
                reachable_nodes.add(node)
                print("  " * level + node.label)
                if len(node.children) != 0:
                    for child in node.children:
                        iterate_graph(child, level + 1)

        reachable_nodes: Set[Node] = set()
        for node in self.roots:
            iterate_graph(node, 0)

    def parent_list(self, node: str) -> List[str]:
        parents = []
        current_node = self.node(node)
        while True:
            if len(current_node.parents) == 0:
                return parents
            else:
                parents.append(current_node.label)
                current_node = current_node.parents[0]

    def distance_of(self, left_node: str, right_node: str) -> int:
        if left_node == right_node:
            return 0
        distance = 1
        reachable_nodes = {left_node}
        while True:
            for node in reachable_nodes.copy():
                for parent in self.node(node).parents:
                    if parent.label not in reachable_nodes:
                        if parent.label == right_node:
                            return distance
                        reachable_nodes.add(parent.label)
                for child in self.node(node).children:
                    if child.label not in reachable_nodes:
                        if child.label == right_node:
                            return distance
                        reachable_nodes.add(child.label)
            distance += 1

    # def distance_of(self, left_node: str, right_node: str) -> int:
    #     left_parent_list = self.parent_list(left_node)
    #     right_parent_list = self.parent_list(right_node)
    #     shared_parents = set(left_parent_list).intersection(set(right_parent_list))
    #     distance = 0
    #     for index, parent in enumerate(left_parent_list):
    #         if parent in shared_parents:
    #             distance += index
    #             break
    #     for index, parent in enumerate(right_parent_list):
    #         if parent in shared_parents:
    #             distance += index
    #             break
    #     return distance

    def unique_parent_distance(self, node: str) -> int:
        distence = 0
        current_node = self.node(node)
        while True:
            if len(current_node.parents) == 1:
                distence += 1
                current_node = current_node.parents[0]
            elif len(current_node.parents) == 0:
                return distence, True
            else:
                return distence, False


def imagenet_class_tree() -> IOAction[Tree]:
    def get_class_tree() -> Tree:
        def build_whole_tree() -> Tree:
            with open(abspath("wordnet.is_a.txt"), "r") as file:
                lines = file.readlines()
            tree = Tree()
            for line in lines:
                parent, child = line[:-1].split(" ")
                tree.add_edge(parent, child)
            return tree

        def find_root(tree: Tree):
            visited_nodes = set()

            def find_root_from_node(node: Node):
                if node not in visited_nodes:
                    visited_nodes.add(node)
                    if len(node.parents) == 0:
                        tree.roots.append(node)
                    else:
                        for parent in node.parents:
                            find_root_from_node(parent)

            for node in tree.nodes():
                find_root_from_node(node)

        def remove_unused_nodes(tree: Tree):
            with open(abspath("imagenet_lsvrc_2015_synsets.txt"), "r") as file:
                lines = file.readlines()
            imagenet_labels = list(map(lambda line: line[:-1], lines))
            tree.imagenet_labels = imagenet_labels

            def iterate_graph(node: Node):
                if node not in reachable_nodes:
                    reachable_nodes.add(node)
                    if len(node.parents) != 0:
                        for parent in node.parents:
                            iterate_graph(parent)

            reachable_nodes: Set[Node] = set()
            for label in tree.imagenet_labels:
                iterate_graph(tree.node(label))
            for node in tree.label_to_node.copy().values():
                for child in node.children.copy():
                    if child not in reachable_nodes:
                        node.children.remove(child)
                for parent in node.parents.copy():
                    if parent not in reachable_nodes:
                        node.parents.remove(parent)
                if node not in reachable_nodes:
                    del tree.label_to_node[node.label]
                    if node in tree.roots:
                        tree.roots.remove(node)

        def count_class_num(tree: Tree, level: int):
            count = [0]
            reachable_nodes: Set[Node] = set()

            def iterate_graph(node: Node, current_level: int):
                # if node in reachable_nodes:
                #     print()
                if node not in reachable_nodes:
                    reachable_nodes.add(node)
                else:
                    return
                if current_level != level:
                    for child in node.children:
                        iterate_graph(child, current_level + 1)
                else:
                    count[0] += 1

            iterate_graph(tree.roots[0], 0)
            return count[0]

        def get_parents(tree: Tree, level: int) -> Set[Node]:
            reachable_nodes: Set[Node] = set()

            def iterate_graph(node: Node, current_level: int):
                if current_level != level:
                    for parent in node.parents:
                        iterate_graph(parent, current_level + 1)
                else:
                    reachable_nodes.add(node)

            for label in tree.imagenet_labels:
                iterate_graph(tree.node(label), 0)
            return reachable_nodes

        tree = build_whole_tree()
        find_root(tree)
        # print(max([len(node.children) for node in tree.nodes()]))
        # tree.print()
        remove_unused_nodes(tree)
        # print(count_class_num(tree, 5))
        # tree.print()
        # parents = get_parents(tree, 3)
        # class_label = tree.imagenet_labels[100]
        # for label in tree.imagenet_labels:
        #     print(tree.distance_of(class_label, label))
        return tree

    path = "store/class_hierarchy/imagenet.pkl"
    return IOAction(path, init_fn=get_class_tree, cache=True)


if __name__ == "__main__":
    # imagenet_class_tree().save()
    tree = imagenet_class_tree().load()
    class_label = tree.imagenet_labels[0]
    for label in tree.imagenet_labels:
        print(tree.distance_of(class_label, label))
