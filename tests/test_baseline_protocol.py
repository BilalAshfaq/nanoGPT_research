import os
import unittest


class BaselineProtocolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        repository_root = os.path.dirname(os.path.dirname(__file__))
        protocol_path = os.path.join(
            repository_root,
            'tuned_global_sgdm',
            'Baseline_Tuning_Protocol.md',
        )
        with open(protocol_path, encoding='utf-8') as protocol_file:
            cls.protocol = protocol_file.read()

    def test_candidate_grids_and_run_limits_are_explicit(self):
        required_text = (
            'matrix_learning_rate in {0.03, 0.10, 0.30, 1.00}',
            'matrix_momentum in {0.90, 0.95, 0.99}',
            'learning_rate in {3e-4, 6e-4, 1e-3, 2e-3}',
            'beta1 in {0.85, 0.90, 0.95}',
            'Current approved-manifest ceiling | 24 | 4 | 28',
            'future Full Muon budget of 12 exploratory plus 2 additional',
        )
        for text in required_text:
            with self.subTest(text=text):
                self.assertIn(text, self.protocol)

    def test_selection_budget_and_confirmation_seeds_are_fixed(self):
        required_text = (
            'selection_tokens = 999 * 491,520 = 491,028,480',
            'hard per-run limit is 1,000',
            '1,337; 2,027; 4,099',
            'sample standard deviation',
        )
        for text in required_text:
            with self.subTest(text=text):
                self.assertIn(text, self.protocol)

    def test_matched_controls_and_authorization_boundary_are_documented(self):
        required_text = (
            'does not authorize or\nlaunch any tuning run',
            'identical model architecture and initialization seed',
            'identical sampled data order for each shared seed',
            'identical selection and maximum processed-token budgets',
            'Any mismatch blocks all\ntuning runs',
        )
        for text in required_text:
            with self.subTest(text=text):
                self.assertIn(text, self.protocol)


if __name__ == '__main__':
    unittest.main()
