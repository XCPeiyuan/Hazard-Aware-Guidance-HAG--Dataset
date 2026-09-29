import json
import sys
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import mr05_stats as s

class StatisticsTests(unittest.TestCase):

    def test_exact_normalization_and_quality(self):
        self.assertEqual(s.classify_response(' \n<SAFE/> '), 'safe')
        for text in ('', '<SAFE />', 'prefix<SAFE/>', '<SAFE/> extra'):
            self.assertEqual(s.classify_response(text), 'hazard')
            self.assertTrue(s.quality_flags(text))
        self.assertEqual(s.quality_flags('<ALERT>Obstacle</ALERT><GUIDE>Turn</GUIDE>'), [])
        self.assertIn('empty_output', s.quality_flags(' '))

    def test_percentile_interpolates_and_empty_is_null(self):
        self.assertAlmostEqual(s.percentile([0, 10], 0.95), 9.5)
        self.assertIsNone(s.percentile([], 0.95))
        with self.assertRaises(ValueError):
            s.percentile([1], 95)

    def test_pool_observations_preserve_duplicate_images_and_errors(self):
        rows = [dict(status='ok', record_id=str(i), image_name='same.png', response_text='<SAFE/>', end_to_end_latency_sec=v, generated_tokens=4) for i, v in enumerate([1, 2, 3, 100])]
        rows += [dict(status='error', error='failed', response_text='', end_to_end_latency_sec=None)]
        result = s.summarize_rows(rows)
        self.assertEqual(result['successes'], 4)
        self.assertEqual(result['failures'], 1)
        self.assertAlmostEqual(result['branches']['safe']['p95_end_to_end_latency_sec'], 85.45)
        self.assertIsNone(result['branches']['hazard']['mean_end_to_end_latency_sec'])
        self.assertEqual(result['branches']['safe']['count'], 4)

    def test_absent_run_cannot_pass(self):
        result = s.validate_run(Path(__file__).resolve().parent / 'nonexistent_run')
        self.assertFalse(result['passed'])
        self.assertFalse(result['complete'])
        self.assertTrue(result['errors'])
if __name__ == '__main__':
    unittest.main()
