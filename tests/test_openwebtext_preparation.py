import ast
from pathlib import Path
import re
import unittest


REPO_ROOT = Path(__file__).resolve().parents[1]
PREPARE_SCRIPT = REPO_ROOT / "data" / "openwebtext" / "prepare.py"
PREPARATION_JOB = REPO_ROOT / "prepare_openwebtext.slurm"


class OpenWebTextPreparationTests(unittest.TestCase):
    def test_dataset_source_and_revision_are_pinned(self):
        tree = ast.parse(PREPARE_SCRIPT.read_text(encoding="utf-8"))
        calls = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "load_dataset"
        ]

        self.assertEqual(len(calls), 1)
        call = calls[0]
        self.assertEqual(ast.literal_eval(call.args[0]), "Skylion007/openwebtext")
        keywords = {keyword.arg: keyword.value for keyword in call.keywords}
        self.assertEqual(
            ast.literal_eval(keywords["revision"]),
            "79d93d786212f7344586290adb811d4ae6a1762c",
        )

    def test_pinned_pipeline_token_counts_preserve_historical_total(self):
        script = PREPARATION_JOB.read_text(encoding="utf-8")

        def shell_integer(name):
            match = re.search(rf"^{name}=(\d+)$", script, flags=re.MULTILINE)
            self.assertIsNotNone(match, f"missing {name}")
            return int(match.group(1))

        train_tokens = shell_integer("EXPECTED_TRAIN_TOKENS")
        val_tokens = shell_integer("EXPECTED_VAL_TOKENS")
        self.assertEqual(train_tokens, 9_035_582_489)
        self.assertEqual(val_tokens, 4_434_606)
        self.assertEqual(train_tokens + val_tokens, 9_040_017_095)


if __name__ == "__main__":
    unittest.main()
