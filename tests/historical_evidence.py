"""Opt-in checks of private historical records, NOT portable unit tests.

Set NEUROSTATION_VERIFY_HISTORICAL_EVIDENCE=1 only in the original authorized
evidence workspace. Missing or changed artifacts then FAIL normally. No data is
uploaded or fabricated, and a skipped check is never historical acceptance.
"""
import os
import unittest

historical_evidence = unittest.skipUnless(
    os.environ.get("NEUROSTATION_VERIFY_HISTORICAL_EVIDENCE") == "1",
    "private historical evidence: opt in with NEUROSTATION_VERIFY_HISTORICAL_EVIDENCE=1; not verified by portable CI",
)
