"""Extract medication names and optional doses with a local classifier checkpoint."""
import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import medication_attributes, medication_doses, medication_regimens, name_spans, token_classifier


class MedicationNameExtractor:
    def __init__(self, run, device="cpu", attribute_run=None, validate_attributes=True, attribute_candidates=False):
        import torch
        from transformers import AutoModelForTokenClassification, AutoTokenizer
        if device not in ('cpu', 'cuda'):
            raise ValueError('Device must be cpu or cuda')
        self.device = device
        self.validate_attributes=validate_attributes
        self.candidate_attributes=attribute_candidates
        if attribute_run and attribute_candidates:
            raise ValueError('Choose a model or candidates for attributes')
        run = Path(run)
        self.report = json.loads((run/'run.json').read_text())
        classifiers={m.VERSION:m for m in (token_classifier,medication_doses,medication_regimens,medication_attributes)}
        self.classifier=classifiers.get(self.report['alignment_version'])
        if (self.classifier is None
                or self.report['source_token_version'] != name_spans.VERSION):
            raise ValueError('Classifier source alignment version differs from training')
        self.config = self.report['config']
        torch.set_num_threads(self.report['threads'])
        self.tokenizer = AutoTokenizer.from_pretrained(run, local_files_only=True, use_fast=True)
        self.model = AutoModelForTokenClassification.from_pretrained(run, local_files_only=True)
        if self.model.config.id2label != dict(enumerate(self.classifier.LABELS)):
            raise ValueError('Saved model labels differ from BIO schema')
        if attribute_candidates and self.classifier is not medication_doses:
            raise ValueError('Attribute candidates require a name/dose checkpoint')
        self.model.to(self.device).eval()
        self.attributes=None
        if attribute_run is not None:
            if self.classifier is not medication_doses:
                raise ValueError('Separate attribute inference requires a name/dose base checkpoint')
            self.attributes=MedicationNameExtractor(attribute_run,device=device)
            if self.attributes.classifier is not medication_attributes:
                raise ValueError('Expected a separate route/frequency checkpoint')

    def predict_payload(self, text):
        import torch
        if not isinstance(text, str) or not text.strip():
            raise ValueError('Provide nonempty note text')
        started = time.monotonic()
        encoded, words = self.classifier.encode_record(
            {'text': text}, self.tokenizer, self.config['max_length'], labeled=False)
        with torch.inference_mode(), torch.autocast(
                self.device, dtype=torch.bfloat16, enabled=self.config['compute_dtype'] == 'bfloat16'):
            logits = self.model(**{k: torch.tensor([v], device=self.device) for k, v in encoded.items()}).logits
        payload = self.classifier.decode_predictions(
            words, logits.argmax(-1)[0].tolist(), len(name_spans.source_tokens(text)))
        if self.attributes is not None:
            payload=medication_attributes.merge_payload(payload,self.attributes.predict_payload(text)[0])
        if self.candidate_attributes:
            payload=medication_regimens.candidate_payload(text,payload)
        if self.validate_attributes and 'dose_spans' in payload and 'route_spans' in payload:
            payload=medication_regimens.validated_payload(text,payload)
        return payload, time.monotonic()-started

    def predict(self, text):
        if self.classifier is medication_attributes:
            raise ValueError('Use this checkpoint with a dose model through --attribute-model')
        payload, elapsed = self.predict_payload(text)
        result = (medication_regimens.output(text,payload) if 'route_spans' in payload else
                  medication_doses.output(text, payload) if 'dose_spans' in payload else
                  {'medication_names': name_spans.resolve_response({'text': text}, json.dumps(payload))})
        if 'route_spans' in payload:
            result['attribute_extraction']='source-candidates-v1' if self.candidate_attributes else 'bio-classifier-v1'
        return dict(result, elapsed_seconds=elapsed, offset_unit='unicode_code_points', end_exclusive=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path, help='Saved classifier directory containing run.json')
    parser.add_argument('--input', type=Path, help='UTF-8 note file; defaults to stdin')
    attributes=parser.add_mutually_exclusive_group()
    attributes.add_argument('--attribute-model',type=Path,help='Separate route/frequency checkpoint')
    attributes.add_argument('--attribute-candidates',action='store_true',help='Use source route/frequency candidates')
    args = parser.parse_args()
    try:
        text = args.input.read_text(encoding='utf-8') if args.input else sys.stdin.read()
        result = MedicationNameExtractor(args.run,attribute_run=args.attribute_model,attribute_candidates=args.attribute_candidates).predict(text)
    except (ValueError, OSError, KeyError) as error:
        parser.exit(2, f'Error: {error}\n')
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
