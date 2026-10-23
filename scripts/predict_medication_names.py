"""Extract medication names from UTF-8 text with a local classifier checkpoint."""
import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import name_spans, token_classifier


class MedicationNameExtractor:
    def __init__(self, run):
        import torch
        from transformers import AutoModelForTokenClassification, AutoTokenizer
        run = Path(run)
        self.report = json.loads((run/'run.json').read_text())
        if (self.report['alignment_version'] != token_classifier.VERSION
                or self.report['source_token_version'] != name_spans.VERSION):
            raise ValueError('Classifier source alignment version differs from training')
        self.config = self.report['config']
        torch.set_num_threads(self.report['threads'])
        self.tokenizer = AutoTokenizer.from_pretrained(run, local_files_only=True, use_fast=True)
        self.model = AutoModelForTokenClassification.from_pretrained(run, local_files_only=True)
        if self.model.config.id2label != dict(enumerate(token_classifier.LABELS)):
            raise ValueError('Saved model labels differ from BIO schema')
        self.model.eval()

    def predict_payload(self, text):
        import torch
        if not isinstance(text, str) or not text.strip():
            raise ValueError('Provide nonempty note text')
        started = time.monotonic()
        encoded, words = token_classifier.encode_record(
            {'text': text}, self.tokenizer, self.config['max_length'], labeled=False)
        with torch.inference_mode(), torch.autocast(
                'cpu', dtype=torch.bfloat16, enabled=self.config['compute_dtype'] == 'bfloat16'):
            logits = self.model(**{k: torch.tensor([v]) for k, v in encoded.items()}).logits
        payload = token_classifier.decode_predictions(
            words, logits.argmax(-1)[0].tolist(), len(name_spans.source_tokens(text)))
        return payload, time.monotonic()-started

    def predict(self, text):
        payload, elapsed = self.predict_payload(text)
        return {'medication_names': name_spans.resolve_response({'text': text}, json.dumps(payload)),
                'elapsed_seconds': elapsed,
                'offset_unit': 'unicode_code_points', 'end_exclusive': True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path, help='Saved classifier directory containing run.json')
    parser.add_argument('--input', type=Path, help='UTF-8 note file; defaults to stdin')
    args = parser.parse_args()
    try:
        text = args.input.read_text(encoding='utf-8') if args.input else sys.stdin.read()
        result = MedicationNameExtractor(args.run).predict(text)
    except (ValueError, OSError, KeyError) as error:
        parser.exit(2, f'Error: {error}\n')
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
