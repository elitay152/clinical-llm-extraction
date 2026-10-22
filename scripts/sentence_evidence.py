"""Map numbered source segments to exact Unicode evidence spans."""
import json
import re

from scripts.annotation_data import FIELDS

MODE = 'sentence_ids'
VERSION = 'source-sentences-v1'
ABBREVIATIONS = {'dr', 'mr', 'mrs', 'ms', 'prof', 'vs', 'etc', 'fig', 'no', 'st', 'jr', 'sr', 'mg', 'mcg', 'ml', 'iv', 'po', 'bid', 'tid'}


def source_sentences(text):
    boundaries = []
    for match in re.finditer(r'\n+|[.!?]+["\u201d\u2019\')\]]*(?=\s|$)', text):
        if match.group().startswith('.'):
            prefix = text[:match.start()]
            word = re.search(r'([\w.]+)$', prefix)
            token = word.group(1).lower() if word else ''
            if token in ABBREVIATIONS or (len(token) == 1 and token.isalpha()) or '.' in token:
                continue
        boundaries.append(match.end())
    units = []
    start = 0
    for end in boundaries + [len(text)]:
        while start < end and text[start].isspace():
            start += 1
        stop = end
        while stop > start and text[stop-1].isspace():
            stop -= 1
        if start < stop:
            units.append({'id': len(units)+1, 'start': start, 'end': stop, 'text': text[start:stop]})
        start = end
    return units


def render_source(text, mode='quotes'):
    if mode == 'quotes':
        return text
    if mode != MODE:
        raise ValueError('Unsupported evidence mode')
    return '\n'.join(f'[{unit["id"]}] {unit["text"]}' for unit in source_sentences(text))


def selected_spans(text, ids):
    units = source_sentences(text)
    if not isinstance(ids, list) or not ids:
        raise ValueError('Select at least one evidence sentence ID')
    if any(type(i) is not int or i < 1 or i > len(units) for i in ids):
        raise ValueError('Unknown evidence sentence ID')
    if len(set(ids)) != len(ids):
        raise ValueError('Duplicate evidence sentence ID')
    spans = []
    previous = None
    for identifier in sorted(ids):
        unit = units[identifier-1]
        if previous is not None and identifier == previous + 1:
            spans[-1]['end'] = unit['end']
            spans[-1]['text'] = text[spans[-1]['start']:unit['end']]
        else:
            spans.append({key: unit[key] for key in ('start', 'end', 'text')})
        previous = identifier
    return spans


def span_ids(text, spans):
    units = source_sentences(text)
    ids = set()
    for span in spans:
        if (type(span['start']) is not int or type(span['end']) is not int or
                not 0 <= span['start'] < span['end'] <= len(text) or text[span['start']:span['end']] != span['text']):
            raise ValueError('Reference evidence does not match source')
        ids.update(unit['id'] for unit in units if unit['start'] < span['end'] and unit['end'] > span['start'])
    return sorted(ids)


def sentence_payload(record):
    medications = []
    for med in record['medications']:
        item = {key: value for key, value in med.items() if key != 'evidence'}
        item['evidence_sentence_ids'] = span_ids(record['text'], med['evidence'])
        medications.append(item)
    return {'medications': medications,
            'negative_evidence_sentence_ids': span_ids(record['text'], record['negative_evidence'])}


def resolve_response(candidate, response):
    from scripts.pseudo_label import validate_resolved_response
    payload = json.loads(response)
    if not isinstance(payload, dict) or set(payload) != {'medications', 'negative_evidence_sentence_ids'}:
        raise ValueError('Output needs medications and negative_evidence_sentence_ids')
    if not isinstance(payload['medications'], list) or not isinstance(payload['negative_evidence_sentence_ids'], list):
        raise ValueError('Medication records and evidence IDs must be arrays')
    meds = []
    for med in payload['medications']:
        if not isinstance(med, dict) or set(med) != (FIELDS - {'evidence'}) | {'evidence_sentence_ids'}:
            raise ValueError('Medication fields do not match sentence-ID schema')
        converted = {key: value for key, value in med.items() if key != 'evidence_sentence_ids'}
        converted['evidence'] = selected_spans(candidate['text'], med['evidence_sentence_ids'])
        meds.append(converted)
    negatives = payload['negative_evidence_sentence_ids']
    negative_spans = selected_spans(candidate['text'], negatives) if negatives else []
    return validate_resolved_response(candidate, meds, negative_spans)
