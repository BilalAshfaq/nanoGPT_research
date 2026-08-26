import unittest

import torch

from model import GPT, GPTConfig
from shared_utils.matched_preflight import (
    assert_preflights_match,
    capture_matched_preflight,
)
from shared_utils.optimizer_factory import configure_optimizer


def model_factory():
    return GPT(
        GPTConfig(
            block_size=8,
            vocab_size=32,
            n_layer=2,
            n_head=2,
            n_embd=8,
            dropout=0.0,
            bias=False,
        )
    )


def configure(model, optimizer_name):
    return configure_optimizer(
        model=model,
        optimizer_name=optimizer_name,
        device_type='cpu',
        adamw_learning_rate=6e-4,
        adamw_weight_decay=0.1,
        adamw_betas=(0.9, 0.95),
        matrix_learning_rate=0.1,
        matrix_momentum=0.95,
        matrix_weight_decay=0.1,
        auxiliary_learning_rate=6e-4,
        auxiliary_weight_decay=0.1,
        auxiliary_betas=(0.9, 0.95),
    )


def batch_sampler_factory():
    data = torch.arange(256, dtype=torch.long)

    def sample():
        indices = torch.randint(len(data) - 8, (4,))
        inputs = torch.stack([data[index:index + 8] for index in indices])
        targets = torch.stack([data[index + 1:index + 9] for index in indices])
        return inputs, targets

    return sample


class BaselinePreflightTests(unittest.TestCase):
    def test_adamw_and_global_sgdm_start_from_identical_state_and_batches(self):
        adamw = capture_matched_preflight(
            seed=1337,
            model_factory=model_factory,
            optimizer_factory=lambda model: configure(model, 'adamw'),
            batch_sampler_factory=batch_sampler_factory,
            num_batches=8,
        )
        global_sgdm = capture_matched_preflight(
            seed=1337,
            model_factory=model_factory,
            optimizer_factory=lambda model: configure(model, 'global_sgdm'),
            batch_sampler_factory=batch_sampler_factory,
            num_batches=8,
        )

        assert_preflights_match(adamw, global_sgdm)
        self.assertEqual(adamw['num_batches'], 8)

    def test_preflight_detects_data_order_mismatch(self):
        first = capture_matched_preflight(
            seed=1337,
            model_factory=model_factory,
            optimizer_factory=lambda model: configure(model, 'adamw'),
            batch_sampler_factory=batch_sampler_factory,
            num_batches=2,
        )
        second = dict(first)
        second['batch_fingerprints'] = list(first['batch_fingerprints'])
        second['batch_fingerprints'][0] = 'different-batch-fingerprint'

        with self.assertRaisesRegex(AssertionError, 'batches do not match'):
            assert_preflights_match(first, second)


if __name__ == '__main__':
    unittest.main()
