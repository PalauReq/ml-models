import unittest

from mcts import *


class TestMTCS(unittest.TestCase):
    def test_search(self):
        class ConstantModel(Model):
            def __call__(self, s: State) -> tuple[list[float], float]:
                return ([0.25]*4, 0.1)

        model = ConstantModel()
        env = Environment()
        root = MCTNode(parent=None, s=State([0]))
        
        a, next_root = search(root, model, env)

        print(root)
        for child in root.children:
            print(child)

        print(f"Playing action {a} to transition to {next_root}")


if __name__ == "__main__":
    unittest.main()