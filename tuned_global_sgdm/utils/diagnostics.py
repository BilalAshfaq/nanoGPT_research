"""Compatibility exports for tuned global SGDM diagnostics."""

from shared_utils.sgdm_diagnostics import (
    SGDMUpdateDiagnostics,
    append_diagnostic_record,
    initialize_diagnostic_log,
    parse_diagnostic_matrix_names,
    parse_diagnostic_steps,
)


GlobalSGDMDiagnostics = SGDMUpdateDiagnostics

__all__ = (
    "GlobalSGDMDiagnostics",
    "append_diagnostic_record",
    "initialize_diagnostic_log",
    "parse_diagnostic_matrix_names",
    "parse_diagnostic_steps",
)
