import unittest

from train import get_action_from_idx, get_idx_from_action


class TestActionIdx(unittest.TestCase):
    def test_idx_to_action(self):
        self.assertEqual(get_action_from_idx(31),
                         [0, 1, 1, 1, 1, 1])
        self.assertEqual(get_action_from_idx(21),
                         [0, 1, 0, 1, 0, 1])
        self.assertEqual(get_action_from_idx(0),
                         [0, 0, 0, 0, 0, 0])
        self.assertEqual(get_action_from_idx(2),
                         [0, 0, 0, 0, 1, 0])

    def test_action_to_idx(self):
        self.assertEqual(
            get_idx_from_action(
                [0, 1, 1, 1, 1, 1]
            ),
            31
        )

        self.assertEqual(
            get_idx_from_action(
                [0, 1, 0, 1, 0, 1]
            ),
            21
        )

        self.assertEqual(
            get_idx_from_action(
                [0, 0, 0, 0, 0, 0]
            ),
            0
        )

        self.assertEqual(
            get_idx_from_action(
                [0, 0, 0, 0, 1, 0]
            ),
            2
        )


if __name__ == '__main__':
    unittest.main()