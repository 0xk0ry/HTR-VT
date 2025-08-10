import unittest
import torch
from utils import utils


class TestCTCPosteriors(unittest.TestCase):
    def test_gamma_rows_sum_to_one(self):
        torch.manual_seed(0)
        T, C = 12, 8
        log_probs = torch.randn(T, C).log_softmax(-1)
        # Build a simple target of length U=3 using indices in [1..C-1]
        y = torch.tensor([1, 2, 3], dtype=torch.long)
        gamma = utils.ctc_posteriors(log_probs, y, blank=0)
        self.assertEqual(gamma.shape, (T, y.numel()))
        # Rows should sum to ~1 (allow tiny numerical error)
        row_sums = gamma.sum(dim=1)
        self.assertTrue(torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5))


if __name__ == '__main__':
    unittest.main()
