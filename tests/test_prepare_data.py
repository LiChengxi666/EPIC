import sys
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from prepare_data import DAPO_PREFIX, DAPO_SUFFIX, INSTRUCTION, answer_text, normalize_rows


class PrepareDataTests(unittest.TestCase):
    def test_dapo_wrapper_matches_boxed_reward(self):
        question = r'Compute $\frac{2}{1}+2$.'
        source = {'prompt': [{'role': 'user', 'content': DAPO_PREFIX + question + DAPO_SUFFIX}],
                  'reward_model': {'ground_truth': '4'}, 'extra_info': {'index': 'source-id'}}
        rows, _ = normalize_rows([source], {'name': 'train', 'data_source': 'lighteval/MATH', 'split': 'train'})
        self.assertEqual(rows[0]['prompt'], [{'role': 'user', 'content': question + '\n\n' + INSTRUCTION}])
        self.assertEqual(rows[0]['extra_info']['source_id'], 'source-id')
        self.assertEqual(rows[0]['reward_model']['ground_truth'], '4')
        self.assertTrue(source['prompt'][0]['content'].endswith(DAPO_SUFFIX))

    def test_unknown_wrapper_fails(self):
        row = {'prompt': [{'role': 'user', 'content': 'New upstream instructions'}],
               'reward_model': {'ground_truth': '4'}}
        with self.assertRaisesRegex(ValueError, 'wrapper'):
            normalize_rows([row], {'name': 'train'})

    def test_dapo_repeated_source_ids(self):
        row = {'prompt': [{'role': 'user', 'content': DAPO_PREFIX + 'Question' + DAPO_SUFFIX}],
               'reward_model': {'ground_truth': '4'}, 'extra_info': {'index': 'same-id'}}
        spec = {'name': 'train', 'data_source': 'lighteval/MATH', 'split': 'train'}
        rows, n = normalize_rows(iter([row, row]), spec)
        self.assertEqual((len(rows), n), (1, 1))
        with self.assertRaisesRegex(ValueError, 'conflicting'):
            normalize_rows([row, {**row, 'reward_model': {'ground_truth': '5'}}], spec)

    def test_eval_does_not_leak_solution(self):
        row = {'problem': 'Find x.', 'answer': '12', 'solution': 'Secret worked solution'}
        rows, _ = normalize_rows([row], {'name': 'aime24', 'data_source': 'aime24',
                                        'problem_field': 'problem', 'split': 'train'})
        self.assertEqual(rows[0]['prompt'][0]['content'], 'Find x.\n\n' + INSTRUCTION)
        self.assertEqual(rows[0]['reward_model']['ground_truth'], '12')

    def test_amc_duplicates_and_conflicts(self):
        spec = {'name': 'amc23', 'data_source': 'amc23', 'problem_field': 'question', 'split': 'test'}
        a = {'question': 'Find x.', 'answer': '3'}
        rows, n = normalize_rows([a, a], spec)
        self.assertEqual((len(rows), n), (1, 1))
        with self.assertRaisesRegex(ValueError, 'conflicting'):
            normalize_rows([a, {**a, 'answer': '4'}], spec)

    def test_answer_preserves_exact_string(self):
        self.assertEqual(answer_text(r'\frac{1}{3}'), r'\frac{1}{3}')
        self.assertEqual(answer_text('007'), '007')
        self.assertEqual(answer_text(7.0), '7')
        for bad in (None, '', True, 0.5):
            with self.assertRaises(ValueError):
                answer_text(bad)


if __name__ == '__main__':
    unittest.main()
