from __future__ import annotations

import unittest

from bridges.harp import extract_final_answer


class HARPExtractionTests(unittest.TestCase):
    def test_last_complete_box_wins_and_preserves_nested_latex(self):
        answer, source = extract_final_answer(
            "A discarded value is \\boxed{3}.\n"
            "Answer: \\boxed{10\\frac{2}{3}}"
        )
        self.assertEqual(answer, r"10\frac{2}{3}")
        self.assertEqual(source, "last_complete_boxed")

    def test_incomplete_final_box_does_not_hide_previous_complete_box(self):
        answer, source = extract_final_answer(
            r"Earlier \boxed{14}; Answer: \boxed{14"
        )
        self.assertEqual(answer, "14")
        self.assertEqual(source, "last_complete_boxed")

    def test_answer_line_is_second_priority(self):
        answer, source = extract_final_answer(
            "The reasoning contains 123.\nAnswer: x^2 + 2x"
        )
        self.assertEqual(answer, "x^2 + 2x")
        self.assertEqual(source, "last_answer_line")

    def test_unformatted_reasoning_is_not_scored_as_an_answer(self):
        self.assertEqual(
            extract_final_answer("The reasoning contains 123."),
            (None, "missing"),
        )


if __name__ == "__main__":
    unittest.main()
