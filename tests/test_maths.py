import argparse

import torch

import ptychi.maths as pmath
import test_utils as tutils


class TestMaths(tutils.BaseTester):
    
    def test_orthgonalize_gs(self):
        torch.manual_seed(123)
        x = torch.rand(2, 3, 4, 5)
        x = pmath.orthogonalize_gs(x, dim=(-2, -1), group_dim=1)
        prod = torch.sum(x[0, 0] * x[0, 1])
        assert prod < 1e-4


    def test_orthgonalize_svd(self):
        torch.manual_seed(123)
        x = torch.rand(2, 4, 5) + 1j * torch.rand(2, 4, 5)
        x = pmath.orthogonalize_svd(x, dim=(-2, -1), group_dim=0)
        prod = torch.sum(x[0] * x[1].conj())
        assert prod.abs() < 1e-4

        x = torch.rand(2, 3, 4, 5) + 1j * torch.rand(2, 3, 4, 5)
        x = pmath.orthogonalize_svd(x, dim=(-2, -1), group_dim=1)
        prod = torch.sum(x[0, 0] * x[0, 1].conj())
        assert prod.abs() < 1e-4

        x = torch.rand(2, 3, 4, 5) + 1j * torch.rand(2, 3, 4, 5)
        x = pmath.orthogonalize_svd(x, dim=(-2, -1), group_dim=0)
        prod = torch.sum(x[0, 0] * x[1, 0].conj())
        assert prod.abs() < 1e-4


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--generate-gold', action='store_true')
    args = parser.parse_args()
    
    tester = TestMaths()
    tester.setup_method(name="", generate_data=False, generate_gold=args.generate_gold, debug=True)
    tester.test_orthgonalize_gs()
    tester.test_orthgonalize_svd()
