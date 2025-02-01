from __future__ import annotations
from abc import abstractmethod
from dataclasses import dataclass
from math import sqrt
import logging


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# @dataclass
# class State():
#     data: list
#
#     def turn():
#         pass
#
#
# class Environment():
#     def transition(self, s: State, a: int) -> State:
#         return State
#
#
# class Model():
#     @abstractmethod
#     def __call__(self, s: State) -> tuple[list[float], float]:
#         pass
#
#
class MCTNode():
    """Represents a state"""
    def __init__(self, parent: MCTNode | None, a: int | None, s: State, p: float = 0):
        self.n = 0 # visit count
        self.w = 0 # total action-value
        self.q = 0 # mean action-value
        self.p = p # prior probability of selecting the edge

        self.parent = parent
        self.a = a
        self.s = s
        self.children = []


    def __repr__(self) -> str:
        return f"MCTNode(n={self.n}, w={self.w:.5f}, q={self.q:.5f}, p={self.p:.5f}, parent={id(self.parent)}, is_leaf={self.is_leaf()}, b={self.s.board.tolist()})"

    def is_leaf(self) -> bool:
        return len(self.children) == 0
    
    def get_best_child_to_explore(self, c_puct: float = 1) -> MCTNode:
        sqrt_sum_n = sqrt(sum(child.n for child in self.children))
        return max(self.children, key=lambda x: x.q + x.upper_confidence_bound(sqrt_sum_n, c_puct))
    
    def upper_confidence_bound(self, sqrt_sum_n: float, c_puct: float = 1) -> float:
        return c_puct * self.p * sqrt_sum_n / (1 + self.n)
    
    def get_best_child_to_play(self, temperature: float = 1) -> MCTNode:
        sum_n = sum(child.n ** (1 / temperature) for child in self.children)
        return max(self.children, key=lambda x: x.n ** (1 / temperature) / sum_n)

    def get_policy(self, temperature: float = 1) -> list[float]:
        sum_n = sum(child.n ** (1 / temperature) for child in self.children)
        return [child.n ** (1 / temperature) / sum_n for child in self.children]

    def get_action(self) -> int:
        return self.a


def search(root: MCTNode, f: Model, env: Environment, num_simulations: int = 800, temperature: float = 1) -> list[float]:
    for t in range(num_simulations):
        logger.debug(f"simulation: {t}")
        leaf = select(root)
        logger.debug(f"selected leaf: {leaf}")
        expand_and_evaluate(leaf, f, env)
        backup(leaf)

    return root.get_policy(temperature)


def select(node: MCTNode) -> MCTNode:
    while not node.is_leaf():
        node = node.get_best_child_to_explore()
    return node


from tinygrad import Tensor


def expand_and_evaluate(leaf: MCTNode, f: Model, env: Environment):
    # TODO queue nodes for evaluation with batch_size=8
    x = Tensor(leaf.s.board).reshape((1, 2, 3, 3))
    ps, v = f(x) # TODO implement state representation as Tensor
    logger.debug(f"v: {v.item()}, p: {ps.numpy().tolist()}")
    for a, p in enumerate(ps.numpy().flatten()):
        if not env.is_legal(a, leaf.s): continue
        s, _, _ = env.transition(leaf.s, a)
        s.turn()
        child = MCTNode(leaf, a=a, s=s, p=p)
        # logger.debug(f"adding child: {child}")
        leaf.children.append(child)
    leaf.w = v.item()


def backup(leaf: MCTNode):
    v = leaf.w
    leaf.n += 1
    leaf.q = leaf.w / leaf.n
    leaf = leaf.parent
    
    while leaf is not None:
        leaf.n += 1
        leaf.w += v # TODO: consider player POV! node.value_sum += value if node.to_play == to_play else -value
        leaf.q = leaf.w / leaf.n
        leaf = leaf.parent


def play(current_node: MCTNode, temperature: float) -> tuple[int, MCTNode]:
    # TODO use virtual loss to ensure each search thread evaluates different nodes.
    next_node = current_node.get_best_child_to_play(temperature)
    a = next_node.get_action()
    return a, next_node