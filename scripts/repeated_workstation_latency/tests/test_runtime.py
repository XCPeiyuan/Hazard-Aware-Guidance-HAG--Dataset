import sys, unittest, tempfile, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

class RuntimeTests(unittest.TestCase):

    def test_manifest_preserves_duplicate_images_and_round_major_order(self):
        import mr05_common as c
        rs = [{'record_id': 'a', 'run_order': 1, 'image_name': 'same.png', 'source_group': 'safe'}, {'record_id': 'b', 'run_order': 2, 'image_name': 'same.png', 'source_group': 'hazard'}]
        self.assertEqual([(n, r['record_id']) for n, r in c.schedule(rs, 2)], [(1, 'a'), (1, 'b'), (2, 'a'), (2, 'b')])

    def test_warmup_has_ten_alternating_records(self):
        import mr05_common as c
        rs = [{'record_id': f's{i}', 'source_group': 'safe'} for i in range(8)] + [{'record_id': f'h{i}', 'source_group': 'hazard'} for i in range(8)]
        self.assertEqual([r['record_id'] for r in c.warmup_samples(rs)], ['s0', 'h0', 's1', 'h1', 's2', 'h2', 's3', 'h3', 's4', 'h4'])

    def test_timing_sums_and_failure_nulls(self):
        import mr05_common as c
        d = c.timing_fields([10, 11, 13, 16, 20, 25])
        self.assertEqual(d['end_to_end_latency_sec'], 15)
        self.assertEqual(sum((d[k] for k in c.STAGES)), 15)
        self.assertTrue(all((c.failure_result(ValueError('bad'))[k] is None for k in c.STAGES + ['end_to_end_latency_sec'])))
        with self.assertRaises(ValueError):
            c.timing_fields([0, 1, 0.5, 2, 3, 4])

    def test_empty_or_misaligned_manifest_rejected(self):
        import mr05_common as c
        with self.assertRaises(ValueError):
            c.validate_samples([])
        with self.assertRaises(ValueError):
            c.validate_samples([{'record_id': 'x', 'run_order': 2, 'source_group': 'safe', 'image_name': 'a.png'}])

    def test_barrier_abort_and_timeout(self):
        import mr05_common as c
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)
            (p / 'ABORT.json').write_text('{}')
            with self.assertRaises(RuntimeError):
                c.await_release(p, 0)
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(TimeoutError):
                c.await_release(Path(tmp), 0)

    def test_inference_real_boundaries_include_transfer_sync_and_decode(self):
        from PIL import Image
        import mr05_worker as w
        import contextlib
        events = []

        class Tensor:

            def __init__(self, values):
                self.values = values

            def __len__(self):
                return len(self.values)

            def __getitem__(self, x):
                return Tensor(self.values[x]) if isinstance(x, slice) else self.values[x]

            def to(self, device):
                events.append('transfer')
                return self

            @property
            def shape(self):
                return (len(self.values),)

            def tolist(self):
                return self.values

        class Torch:
            Tensor = Tensor if False else None
            no_grad = staticmethod(contextlib.nullcontext)

            class cuda:

                @staticmethod
                def synchronize():
                    events.append('sync')
        Torch.Tensor = Tensor

        class Processor:

            def apply_chat_template(self, messages, **kw):
                events.append('preprocess')
                self.messages = messages
                return {'input_ids': [Tensor([1, 2])], 'pixels': Tensor([7])}

            def batch_decode(self, trimmed, **kw):
                events.append('decode')
                return [' <SAFE/> ']

        class Model:
            device = 'cuda:0'

            class generation_config:
                eos_token_id = [9]

            def generate(self, **kw):
                events.append('generate')
                return [Tensor([1, 2, 8, 9])]
        clock = iter([10.0, 11.0, 13.0, 16.0, 20.0, 25.0])
        proc = Processor()
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / 'test.png'
            Image.new('RGB', (2, 2)).save(p)
            out = w.infer_once(Model(), proc, p, 'PROMPT', 1024, torch_module=Torch, clock=lambda: next(clock))
        self.assertEqual(events, ['preprocess', 'transfer', 'sync', 'generate', 'sync', 'decode'])
        self.assertEqual(out['generated_token_ids'], [8, 9])
        self.assertEqual(out['generated_tokens'], 2)
        self.assertTrue(out['ended_with_eos'])
        self.assertEqual(out['response_text'], '<SAFE/>')
        self.assertEqual(out['end_to_end_latency_sec'], 15)
        self.assertEqual(out['transfer_sec'], 3)
        self.assertEqual(out['decode_sec'], 5)
if __name__ == '__main__':
    unittest.main()
