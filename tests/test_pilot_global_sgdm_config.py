import os
import unittest


class PilotGlobalSGDMConfigTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        repository_root = os.path.dirname(os.path.dirname(__file__))
        config_path = os.path.join(
            repository_root,
            'config',
            'pilot_global_sgdm.py',
        )
        namespace = {}
        with open(config_path, encoding='utf-8') as config_file:
            exec(compile(config_file.read(), config_path, 'exec'), namespace)
        cls.config = namespace

    def test_pilot_is_long_enough_to_show_a_validation_trend(self):
        self.assertEqual(self.config['max_iters'], 500)
        self.assertEqual(self.config['eval_interval'], 100)
        self.assertEqual(self.config['eval_iters'], 50)

    def test_two_gpu_accumulation_and_optimizer_settings_are_explicit(self):
        self.assertEqual(self.config['gradient_accumulation_steps'], 4)
        self.assertEqual(self.config['optimizer_name'], 'global_sgdm')
        self.assertEqual(self.config['matrix_learning_rate'], 0.1)
        self.assertEqual(self.config['matrix_momentum'], 0.95)
        self.assertEqual(self.config['matrix_weight_decay'], 0.1)

    def test_pilot_uses_bfloat16_compilation_and_no_wandb(self):
        self.assertEqual(self.config['dtype'], 'bfloat16')
        self.assertTrue(self.config['compile'])
        self.assertFalse(self.config['wandb_log'])


if __name__ == '__main__':
    unittest.main()
