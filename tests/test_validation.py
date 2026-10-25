"""Input boundaries that must fail before a lossy conversion occurs."""
import unittest

import numpy as np
import torch

from conv_rosa_transformer import rosa_core


class ValidationTests(unittest.TestCase):
    @staticmethod
    def inputs():
        return [[0, 1], [1, 0], np.ones((2, 1), dtype=np.uint8),
                np.ones((2, 1)), np.zeros(1), np.ones(1), 1]

    def test_complex_gradient_and_embeddings_are_rejected(self):
        for position in (3, 4, 5):
            for use_torch in (False, True):
                with self.subTest(position=position, torch=use_torch):
                    inputs = self.inputs()
                    value = np.asarray(inputs[position], dtype=np.complex128) + 1j
                    inputs[position] = torch.from_numpy(value) if use_torch else value
                    with self.assertRaises(ValueError):
                        rosa_core.credits_numpy(*inputs)

    def test_integer_torch_gradient_cannot_truncate_credits(self):
        for dtype in (torch.int64, torch.int32, torch.uint8, torch.bool):
            with self.subTest(dtype=dtype):
                inputs = self.inputs()
                inputs[3] = torch.ones((2, 1), dtype=dtype)
                with self.assertRaises(TypeError):
                    rosa_core.exact_stream_bit_credits(*inputs)

    def test_complex_torch_public_gradient_is_rejected(self):
        inputs = self.inputs()
        inputs[3] = torch.ones((2, 1), dtype=torch.complex64)
        with self.assertRaises(TypeError):
            rosa_core.exact_stream_bit_credits(*inputs)


if __name__ == "__main__":
    unittest.main()
