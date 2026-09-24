import unittest

from bruv.data import LETTERS, fit_options


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


if __name__ == "__main__":
    unittest.main()
