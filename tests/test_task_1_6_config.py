import os
import unittest


class Task16ConfigTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        repository_root = os.path.dirname(os.path.dirname(__file__))
        config_path = os.path.join(
            repository_root,
            'config',
            'task_1_6_baseline.py',
        )
        namespace = {}
        with open(config_path, encoding='utf-8') as config_file:
            exec(compile(config_file.read(), config_path, 'exec'), namespace)
        cls.config = namespace

    def test_model_data_and_two_gpu_batch_controls(self):
        self.assertEqual(
            (self.config['n_layer'], self.config['n_head'], self.config['n_embd']),
            (12, 12, 768),
        )
        self.assertEqual(self.config['dataset'], 'openwebtext')
        self.assertEqual(self.config['tokenizer'], 'gpt2')
        self.assertEqual(self.config['block_size'], 1024)
        self.assertEqual(self.config['batch_size'], 12)
        self.assertEqual(self.config['gradient_accumulation_steps'], 40)

    def test_selection_schedule_and_evaluation_controls(self):
        self.assertEqual(self.config['max_iters'], 999)
        self.assertEqual(self.config['eval_interval'], 333)
        self.assertEqual(self.config['eval_iters'], 200)
        self.assertEqual(self.config['warmup_iters'], 100)
        self.assertEqual(self.config['lr_decay_iters'], 999)
        self.assertEqual(self.config['min_lr'], 6e-5)

    def test_numerics_logging_and_diagnostics_controls(self):
        self.assertEqual(self.config['dtype'], 'bfloat16')
        self.assertTrue(self.config['compile'])
        self.assertEqual(self.config['grad_clip'], 1.0)
        self.assertFalse(self.config['wandb_log'])
        self.assertFalse(self.config['diagnostics_enabled'])


if __name__ == '__main__':
    unittest.main()
