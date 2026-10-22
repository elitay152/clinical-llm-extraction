"""Extract medication names by selecting indexed source tokens."""
import hashlib
import json
import re

MODE = 'name_spans'
VERSION = 'source-tokens-v1'


def source_tokens(text):
    return [{'start': m.start(), 'end': m.end(), 'text': m.group()}
            for m in re.finditer(r"\w+(?:[-’']\w+)*|[^\w\s]", text)]


def render_source(text):
    return ' '.join(f'{i}:{token["text"]}' for i, token in enumerate(source_tokens(text), 1))


def resolve_response(record, response):
    payload = json.loads(response)
    if not isinstance(payload, dict) or set(payload) != {'medication_name_spans'}:
        raise ValueError('Output needs medication_name_spans only')
    pairs = payload['medication_name_spans']
    if not isinstance(pairs, list):
        raise ValueError('Name spans must be an array')
    tokens = source_tokens(record['text'])
    spans = []
    seen = set()
    for pair in pairs:
        if (not isinstance(pair, list) or len(pair) != 2 or any(type(i) is not int for i in pair)
                or not 1 <= pair[0] <= pair[1] <= len(tokens)):
            raise ValueError('Invalid name token range')
        start, end = tokens[pair[0]-1]['start'], tokens[pair[1]-1]['end']
        if (start, end) in seen:
            raise ValueError('Duplicate name span')
        seen.add((start, end))
        spans.append({'start': start, 'end': end, 'text': record['text'][start:end]})
    return spans


def reference_payload(record):
    text = record['text']
    tokens = source_tokens(text)
    starts = {t['start']: i for i, t in enumerate(tokens, 1)}
    ends = {t['end']: i for i, t in enumerate(tokens, 1)}
    pairs = set()
    for med in record['medications']:
        matches = []
        for evidence in med['evidence']:
            if text[evidence['start']:evidence['end']] != evidence['text']:
                raise ValueError('Teacher evidence differs from source')
            for match in re.finditer(re.escape(med['name']), evidence['text']):
                start = evidence['start'] + match.start()
                end = evidence['start'] + match.end()
                if start in starts and end in ends:
                    matches.append((starts[start], ends[end]))
        if not matches:
            raise ValueError('Teacher medication name has no aligned source span')
        pairs.update(matches)
    return {'medication_name_spans': [list(pair) for pair in sorted(pairs)]}


def score(references, predictions):
    if not references:
        raise ValueError('No held-out teacher references')
    by_id = {p['id']: p for p in predictions}
    if len(by_id) != len(predictions) or set(by_id) != {r['id'] for r in references}:
        raise ValueError('Prediction membership differs from reference')
    tp = fp = fn = valid = json_valid = exact = 0
    errors = []
    name_tp = name_fp = name_fn = 0
    for ref in references:
        expected_spans = resolve_response(ref, json.dumps(reference_payload(ref)))
        expected = {(s['start'], s['end']) for s in expected_spans}
        try:
            response = by_id[ref['id']]['response']
            json.loads(response)
            json_valid += 1
            spans = resolve_response(ref, response)
            actual = {(s['start'], s['end']) for s in spans}
            valid += 1
            exact += actual == expected
        except (ValueError, TypeError, KeyError) as error:
            spans = []; actual = set()
            errors.append({'id': ref['id'], 'reason': str(error)})
        tp += len(actual & expected); fp += len(actual - expected); fn += len(expected - actual)
        names = {s['text'] for s in spans}
        gold_names = {s['text'] for s in expected_spans}
        name_tp += len(names & gold_names); name_fp += len(names - gold_names); name_fn += len(gold_names - names)
    def metrics(t, f, n):
        return {'precision': t/(t+f) if t+f else 0.0, 'recall': t/(t+n) if t+n else 0.0,
                'f1': 2*t/(2*t+f+n) if 2*t+f+n else None, 'tp': t, 'fp': f, 'fn': n}
    return {'examples': len(references), 'json_valid': json_valid, 'schema_and_spans_valid': valid,
            'exact_document_agreement': exact/len(references), 'exact_name_spans': metrics(tp, fp, fn),
            'unique_name_text': metrics(name_tp, name_fp, name_fn), 'invalid_response_errors': errors,
            'interpretation': 'Agreement with supplied reference name spans; not clinical accuracy.'}


LAB_NEGATIVES = (
    'No current medications. Sodium 138 mmol/L; potassium 4.2 mmol/L.',
    'Medication list is empty. Creatinine 1.1 mg/dL and glucose 96 mg/dL.',
    'No medication use reported. Hemoglobin 13 g/dL; platelets 250 K/uL.',
    'The patient takes no medications. TSH 2.0 mIU/L and free T4 1.1 ng/dL.',
    'No medications are listed. Phosphorus 2.5 mg/dL; calcium 9.2 mg/dL.',
    'Current medications: none. Alkaline phosphatase 575 U/L and vitamin D 16.3 ng/mL.',
    'Medication history is negative. Blood pressure 120/80 mmHg; pulse 72/min.',
    'The patient denies medication use. Temperature 37 C and oxygen saturation 98%.',
)


def synthetic_negatives():
    return [{'id': f'synthetic:lab-negative:{i}', 'text': text, 'split': 'train',
             'source_group': f'synthetic:lab-negative:{i}', 'label_origin': 'synthetic',
             'input_sha256': hashlib.sha256(text.encode()).hexdigest(), 'medications': []}
            for i, text in enumerate(LAB_NEGATIVES, 1)]
