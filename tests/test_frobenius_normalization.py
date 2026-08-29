import inspect
import math
import unittest

import torch

import frobenius_normalized_sgdm.utils.frobenius_normalization as normalization
from frobenius_normalized_sgdm.utils.frobenius_normalization import (
    FrobeniusNormalizationResult,
    normalize_frobenius_momentum,
)


def frobenius_cosine(first, second):
    first = first.double()
    second = second.double()
    return torch.sum(first * second) / (
        torch.linalg.vector_norm(first) * torch.linalg.vector_norm(second)
    )


class FrobeniusNormalizationTests(unittest.TestCase):
    def test_exact_formula_for_square_tall_and_wide_matrices(self):
        cases = (
            ((2, 2), torch.float32, 1.0),
            ((3, 2), torch.float64, 1.5),
            ((2, 4), torch.float32, 0.75),
        )
        for shape, dtype, shape_factor in cases:
            with self.subTest(
                shape=shape,
                dtype=dtype,
                shape_factor=shape_factor,
            ):
                matrix = torch.arange(
                    1,
                    shape[0] * shape[1] + 1,
                    dtype=dtype,
                ).reshape(shape)
                epsilon = 1e-4

                result = normalize_frobenius_momentum(
                    matrix,
                    epsilon=epsilon,
                    fixed_shape_factor=shape_factor,
                )

                calculation_dtype = (
                    torch.float64 if dtype == torch.float64 else torch.float32
                )
                working = matrix.to(calculation_dtype)
                raw_norm = torch.linalg.vector_norm(working)
                nominal_target = shape_factor * math.sqrt(min(shape))
                denominator = raw_norm + epsilon
                expected = (
                    working * (nominal_target / denominator)
                ).to(dtype)
                expected_norm = nominal_target * raw_norm / denominator

                self.assertIsInstance(result, FrobeniusNormalizationResult)
                torch.testing.assert_close(result.normalized_matrix, expected)
                torch.testing.assert_close(result.raw_frobenius_norm, raw_norm)
                torch.testing.assert_close(result.denominator, denominator)
                torch.testing.assert_close(
                    result.nominal_target_norm,
                    torch.as_tensor(nominal_target, dtype=calculation_dtype),
                )
                torch.testing.assert_close(
                    result.epsilon_adjusted_expected_norm,
                    expected_norm,
                )
                torch.testing.assert_close(
                    result.applied_normalization_multiplier,
                    nominal_target / denominator,
                )
                self.assertEqual(result.retained_rank, min(shape))
                self.assertEqual(result.epsilon, epsilon)
                self.assertEqual(result.fixed_shape_factor, shape_factor)
                self.assertFalse(result.zero_momentum.item())
                self.assertFalse(result.epsilon_dominated.item())
                self.assertAlmostEqual(
                    frobenius_cosine(result.normalized_matrix, matrix).item(),
                    1.0,
                    places=6,
                )

    def test_exact_zero_has_finite_zero_output_and_event_metadata(self):
        matrix = torch.zeros((3, 2), dtype=torch.float32)

        result = normalize_frobenius_momentum(matrix, epsilon=1e-6)

        self.assertTrue(torch.equal(result.normalized_matrix, matrix))
        self.assertTrue(torch.isfinite(result.normalized_matrix).all())
        self.assertEqual(result.raw_frobenius_norm.item(), 0.0)
        self.assertAlmostEqual(result.denominator.item(), 1e-6, places=12)
        self.assertEqual(result.epsilon_adjusted_expected_norm.item(), 0.0)
        self.assertEqual(result.applied_normalization_multiplier.item(), 0.0)
        self.assertTrue(result.zero_momentum.item())
        self.assertTrue(result.epsilon_dominated.item())

    def test_below_at_and_above_epsilon_use_the_additive_denominator(self):
        epsilon = 0.25
        for value, dominated in (
            (epsilon / 2.0, True),
            (epsilon, True),
            (epsilon * 2.0, False),
        ):
            with self.subTest(value=value):
                matrix = torch.tensor([[value]], dtype=torch.float64)
                result = normalize_frobenius_momentum(matrix, epsilon)
                expected = value / (abs(value) + epsilon)

                torch.testing.assert_close(
                    result.normalized_matrix,
                    torch.tensor([[expected]], dtype=torch.float64),
                )
                self.assertEqual(result.epsilon_dominated.item(), dominated)
                self.assertFalse(result.zero_momentum.item())
                self.assertTrue(torch.isfinite(result.normalized_matrix).all())

    def test_near_zero_half_precision_inputs_are_finite_and_use_fp32_scalars(self):
        for dtype, value, epsilon in (
            (torch.float16, 1e-5, 1e-4),
            (torch.bfloat16, 1e-8, 1e-7),
        ):
            with self.subTest(dtype=dtype):
                matrix = torch.tensor(
                    [[value, -value], [value, value]],
                    dtype=dtype,
                    requires_grad=True,
                )
                result = normalize_frobenius_momentum(matrix, epsilon)
                working = matrix.detach().float()
                raw_norm = torch.linalg.vector_norm(working)
                expected = (
                    working
                    * (math.sqrt(2.0) / (raw_norm + epsilon))
                ).to(dtype)

                self.assertEqual(result.normalized_matrix.dtype, dtype)
                self.assertEqual(result.raw_frobenius_norm.dtype, torch.float32)
                self.assertEqual(
                    result.applied_normalization_multiplier.dtype,
                    torch.float32,
                )
                self.assertFalse(result.normalized_matrix.requires_grad)
                self.assertTrue(torch.isfinite(result.normalized_matrix).all())
                self.assertTrue(
                    torch.isfinite(result.applied_normalization_multiplier)
                )
                torch.testing.assert_close(result.normalized_matrix, expected)

    def test_positive_rescaling_is_invariant_when_epsilon_is_negligible(self):
        matrix = torch.tensor(
            [[1.0, -2.0], [3.0, -4.0]],
            dtype=torch.float64,
        )
        first = normalize_frobenius_momentum(matrix, epsilon=1e-12)
        second = normalize_frobenius_momentum(7.0 * matrix, epsilon=1e-12)

        torch.testing.assert_close(
            first.normalized_matrix,
            second.normalized_matrix,
            rtol=1e-12,
            atol=1e-12,
        )

    def test_metadata_contains_only_detached_scalar_tensors(self):
        result = normalize_frobenius_momentum(
            torch.ones((2, 3), requires_grad=True),
            epsilon=1e-6,
        )
        scalar_fields = (
            result.raw_frobenius_norm,
            result.denominator,
            result.nominal_target_norm,
            result.epsilon_adjusted_expected_norm,
            result.applied_normalization_multiplier,
            result.zero_momentum,
            result.epsilon_dominated,
        )

        self.assertTrue(
            all(value.ndim == 0 and not value.requires_grad for value in scalar_fields)
        )

    def test_invalid_inputs_and_settings_are_rejected(self):
        valid = torch.ones((2, 2))
        with self.assertRaisesRegex(TypeError, "torch.Tensor"):
            normalize_frobenius_momentum([[1.0]], 1e-6)
        with self.assertRaisesRegex(ValueError, "two-dimensional"):
            normalize_frobenius_momentum(torch.ones(2), 1e-6)
        with self.assertRaisesRegex(ValueError, "dimensions must be positive"):
            normalize_frobenius_momentum(torch.empty((0, 2)), 1e-6)
        with self.assertRaisesRegex(ValueError, "must use float16"):
            normalize_frobenius_momentum(torch.ones((2, 2), dtype=torch.int64), 1e-6)
        sparse = torch.sparse_coo_tensor(
            indices=torch.tensor([[0], [1]]),
            values=torch.tensor([1.0]),
            size=(2, 2),
        )
        with self.assertRaisesRegex(ValueError, "dense strided"):
            normalize_frobenius_momentum(sparse, 1e-6)

        invalid_scalars = (
            0.0,
            -1.0,
            float("inf"),
            float("nan"),
            True,
            "1e-6",
        )
        for value in invalid_scalars:
            with self.subTest(epsilon=value):
                with self.assertRaisesRegex(ValueError, "epsilon must be"):
                    normalize_frobenius_momentum(valid, value)
            with self.subTest(shape_factor=value):
                with self.assertRaisesRegex(ValueError, "fixed_shape_factor must be"):
                    normalize_frobenius_momentum(
                        valid,
                        1e-6,
                        fixed_shape_factor=value,
                    )
        with self.assertRaisesRegex(ValueError, "representable"):
            normalize_frobenius_momentum(valid, 1e-300)
        with self.assertRaisesRegex(ValueError, "too small"):
            normalize_frobenius_momentum(
                valid,
                torch.finfo(torch.float32).tiny,
            )
        with self.assertRaisesRegex(ValueError, "target norm"):
            normalize_frobenius_momentum(
                valid,
                1e-6,
                fixed_shape_factor=1e300,
            )

    def test_implementation_has_no_geometry_transform_dependency(self):
        source = inspect.getsource(normalization).lower()
        for forbidden in ("svd", "polar", "newton", "muon"):
            with self.subTest(forbidden=forbidden):
                self.assertNotIn(forbidden, source)


if __name__ == "__main__":
    unittest.main()
