import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from bruv.data import LETTERS, encode_file, fit_options


class FitOptionsTest(unittest.TestCase):
    def test_wide_task_keeps_gold_and_stable_relabeling(self):
        options = [
            {"label": chr(65 + index), "key": str(index), "description": str(index)}
            for index in range(24)
        ]
        record = {"id": "wide-1", "options": options, "answer": "X"}
        adapted = fit_options(record)
        self.assertEqual(adapted, fit_options(record))
        self.assertEqual(
            [option["label"] for option in adapted["options"]], list(LETTERS)
        )
        self.assertEqual(
            adapted["options"][LETTERS.index(adapted["answer"])]["key"], "23"
        )
        self.assertEqual(record["answer"], "X")

    def test_encoded_slices_skip_records_before_the_limit(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "records.jsonl"
            path.write_text(''.join(f'{{"id": {index}}}\n' for index in range(5)))
            with patch("bruv.data.encode_record", side_effect=lambda record, *_: record):
                self.assertEqual(
                    [row["id"] for row in encode_file(path, None, 1, limit=2, offset=2)],
                    [2, 3],
                )


if __name__ == "__main__":
    unittest.main()
