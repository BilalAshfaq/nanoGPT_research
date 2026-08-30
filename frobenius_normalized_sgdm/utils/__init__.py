"""Variant-local utilities for Frobenius-normalized SGDM."""

from frobenius_normalized_sgdm.utils.frobenius_normalization import (
    FrobeniusNormalizationResult,
    normalize_frobenius_momentum,
)
from frobenius_normalized_sgdm.utils.frobenius_normalized_sgdm import (
    FROBENIUS_NORMALIZATION_EQUATION,
    FROBENIUS_NORMALIZATION_VERSION,
    FrobeniusNormalizedSGDM,
)
from frobenius_normalized_sgdm.utils.diagnostics import (
    FrobeniusNormalizedSGDMDiagnostics,
)


__all__ = (
    "FrobeniusNormalizationResult",
    "FrobeniusNormalizedSGDM",
    "FrobeniusNormalizedSGDMDiagnostics",
    "FROBENIUS_NORMALIZATION_EQUATION",
    "FROBENIUS_NORMALIZATION_VERSION",
    "normalize_frobenius_momentum",
)
