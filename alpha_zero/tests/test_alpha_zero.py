import unittest

from alpha_zero import ResNet, optimize

from tinygrad import Tensor


class TestAlphaZero(unittest.TestCase):
    def test_network_forward_pass_with_batch_size_1(self):
        x = Tensor.rand((1, 17, 19, 19))
        network = ResNet()

        network(x)

    def test_optimize(self):
        import environments.tictactoe as env
        import random

        def simulate_game() -> tuple[list[env.State], list[int], list[float]]:
            is_term = False
            s = env.State()
            states = []
            actions = []
            while not is_term:
                states.append(s.board)

                legal_actions = env.get_legal_actions(s)
                a = random.choice(legal_actions)
                actions.append(a)

                s, r, is_term = env.transition(s, a)
                s.turn()

            if r == 0:
                values = [0 for _ in states]
            else:
                values = []
                for i in range(len(states)):
                    values.append(r)
                    r = -r
                values = values[::-1]

            return states, actions, values

        x, pi, v = [], [], []
        for i in range(100):
            states, actions, values = simulate_game()
            ps = [[1 if a==i else 0 for i in range(3*3+1)] for a in actions]
            x.extend(states), pi.extend(ps), v.extend(values)

        device = "python"
        x, pi, v = Tensor(x, device=device), Tensor(pi, device=device), Tensor(v, device=device)
        r = ResNet((3, 3), 2, 8, 2)
        f = optimize(r, (x, pi, v), num_steps=100, batch_size=8)

if __name__ == "__main__":
    unittest.main()