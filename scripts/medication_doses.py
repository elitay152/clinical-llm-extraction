"""Source-aligned dose entities, conservative linking, and exact-span scoring."""
import json
import re

from scripts import name_spans, token_classifier

VERSION = 'medication-name-dose-bio-v1'
PROTOCOL = 'medication-dose-candidates-v1'
LABELS = token_classifier.LABELS + ['B-DOSE', 'I-DOSE']
LINKING_RULE = 'conservative-local-rule-v2'
LINKING_RULES = ('conservative-local-rule-v1', LINKING_RULE)
_ROUTE = r'(?:oral(?:ly)?|intravenous(?:ly)?|IV|intramuscular(?:ly)?|IM|subcutaneous(?:ly)?|SC|inhaled|topical(?:ly)?)'
_DOSE_INTRO = (r'(?:(?:at|in|with)\s+(?:(?:a|the)\s+)?(?:(?:starting|initial|maximal)\s+)?'
               r'(?:dose|dosage)(?:\s+of)?|(?:dose|dosage)\s+(?:reduced\s+to|was|is)|'
               r'dose(?:\s+of)?|at|to|is|was|of)')
_AFTER_NAME = re.compile(r'\s*[,:(=]?\s*(?:'+_ROUTE+r'\s+)?(?:'+_DOSE_INTRO+r'\s*)?', re.I)
_BEFORE_NAME = re.compile(r'\s*(?:of\s+)?(?:(?:once\s+)?daily\s+)?(?:'+_ROUTE+r'\s+)?\(?\s*', re.I)

CANDIDATE_VERSION = 'dose-amount-candidates-v1'
_AMOUNT = r'(?:\d+(?:\.\d+)?(?:\s*[-–/]\s*\d+(?:\.\d+)?)?|one|two|three|four|five|six|half|quarter)'
_UNIT = r'(?:mcg|[µμu]g|mg|grams?|g|m[lL]|units?|IU|mEq|mmol|tablets?|capsules?|puffs?|drops?|patch(?:es)?|sprays?)'
CANDIDATE = re.compile(r'(?<![\w.])'+_AMOUNT+r'\s*'+_UNIT+r'(?:\s*/\s*(?:kg|m2|m²))?(?![\w])', re.I)


def dose_candidates(text):
    tokens = name_spans.source_tokens(text)
    starts = {t['start'] for t in tokens}; ends = {t['end'] for t in tokens}
    return [{'start':m.start(),'end':m.end(),'text':m.group()} for m in CANDIDATE.finditer(text)
            if m.start() in starts and m.end() in ends
            and not re.match(r'\s*/\s*(?:ml|l|g|h(?:ou)?rs?|days?|min)\b', text[m.end():], re.I)]


def marked_source(record):
    names = name_spans.resolve_response(record,json.dumps(name_spans.reference_payload(record)))
    candidates = dose_candidates(record['text'])
    spans = [(s['start'],s['end'],f'MED_{i}') for i,s in enumerate(names)]
    for i,span in enumerate(candidates):
        if any(span['start'] < n['end'] and n['start'] < span['end'] for n in names):
            raise ValueError('Dose candidate overlaps a frozen medication name')
        spans.append((span['start'],span['end'],f'DOSE_{i}'))
    cursor=0; pieces=[]
    for start,end,label in sorted(spans):
        pieces.extend([record['text'][cursor:start], f'⟦{label}:', record['text'][start:end], '⟧'])
        cursor=end
    pieces.append(record['text'][cursor:])
    return ''.join(pieces), names, candidates


def reference_spans(record):
    if record.get('dose_label_protocol') != PROTOCOL:
        raise ValueError('Missing or incompatible dose-label protocol')
    text = record['text']
    tokens = name_spans.source_tokens(text)
    starts = {token['start']: i+1 for i, token in enumerate(tokens)}
    ends = {token['end']: i+1 for i, token in enumerate(tokens)}
    names = name_spans.resolve_response(record, json.dumps(name_spans.reference_payload(record)))
    occupied = {i for start, end in name_spans.reference_payload(record)['medication_name_spans']
                for i in range(start, end+1)}
    pairs = []
    previous = -1
    for span in record['dose_mentions']:
        start, end = span['start'], span['end']
        if (type(start) is not int or type(end) is not int or start not in starts or end not in ends
                or start < previous or start >= end or text[start:end] != span['text']):
            raise ValueError('Dose spans must be sorted, disjoint, exact, and token aligned')
        pair = [starts[start], ends[end]]
        if occupied & set(range(pair[0], pair[1]+1)):
            raise ValueError('Dose and medication entities overlap')
        pairs.append(pair)
        previous = end
    seen = set()
    for relation in record['dose_relations']:
        if not isinstance(relation, dict) or set(relation) != {'medication_index', 'dose_index'}:
            raise ValueError('Invalid dose relation schema')
        med, dose = relation['medication_index'], relation['dose_index']
        if (type(med) is not int or type(dose) is not int or not 0 <= med < len(names)
                or not 0 <= dose < len(pairs) or (med, dose) in seen):
            raise ValueError('Invalid or duplicate dose relation')
        seen.add((med, dose))
    return pairs


def word_labels(record):
    labels = token_classifier.word_labels(record)
    for start, end in reference_spans(record):
        labels[start-1] = 3
        labels[start:end] = [4]*(end-start)
    return labels


def encode_record(record, tokenizer, max_length=512, labeled=True):
    return token_classifier.encode_record(record, tokenizer, max_length, labeled,
                                          label_function=word_labels)


def decode_predictions(word_ids, predictions, word_count):
    if any(type(label) is not int or not 0 <= label < len(LABELS) for label in predictions):
        raise ValueError('Unknown name/dose BIO label')
    names = token_classifier.decode_predictions(word_ids, [p if p < 3 else 0 for p in predictions], word_count)
    doses = token_classifier.decode_predictions(word_ids, [p-2 if p >= 3 else 0 for p in predictions], word_count)
    return dict(names, dose_spans=doses['medication_name_spans'])


def resolve(text, payload):
    record = {'text': text}
    names = name_spans.resolve_response(record, json.dumps({'medication_name_spans': payload['medication_name_spans']}))
    doses = name_spans.resolve_response(record, json.dumps({'medication_name_spans': payload['dose_spans']}))
    return names, doses


def link_doses(text, names, doses, rule=LINKING_RULE):
    if rule not in LINKING_RULES:
        raise ValueError('Unknown dose linking rule')
    relations = []
    for dose_index, dose in enumerate(doses):
        candidates = []
        for index, name in enumerate(names):
            if name['end'] <= dose['start']:
                gap = text[name['end']:dose['start']]
                allowed = (re.fullmatch(r'\s*(?:(?:at|to|dose(?: of)?|is|was|:|=|\()\s*)?', gap, re.I)
                           if rule == 'conservative-local-rule-v1' else _AFTER_NAME.fullmatch(gap))
                if len(gap) <= (40 if rule == 'conservative-local-rule-v1' else 64) and not re.search(r'[\n\r;.!?]', gap) and allowed:
                    candidates.append(index)
            elif dose['end'] <= name['start']:
                gap = text[dose['end']:name['start']]
                allowed = (re.fullmatch(r'\s*(?:of\s+)?', gap, re.I)
                           if rule == 'conservative-local-rule-v1' else _BEFORE_NAME.fullmatch(gap))
                maximum = 20 if rule == 'conservative-local-rule-v1' else 40
                boundary = '\n' in gap if rule == 'conservative-local-rule-v1' else bool(re.search(r'[\n\r;.!?]', gap))
                if len(gap) <= maximum and not boundary and allowed:
                    candidates.append(index)
        if len(candidates) != 1:
            continue
        index = candidates[0]
        # A bare coordinated list can share a dose; abstain instead of choosing its last name.
        if index and names[index-1]['end'] <= names[index]['start'] < dose['start']:
            bridge = text[names[index-1]['end']:names[index]['start']]
            if re.fullmatch(r'\s*(?:,|and|or|and/or)\s*', bridge, re.I):
                continue
        relations.append({'medication_index': index, 'dose_index': dose_index})
    return relations


def output(text, payload):
    names, doses = resolve(text, payload)
    relations = link_doses(text, names, doses)
    linked = {r['dose_index'] for r in relations}
    return {'medication_names': names, 'dose_spans': doses,
            'medications': [{'name': name, 'doses': [doses[r['dose_index']] for r in relations
                                                   if r['medication_index'] == index]}
                            for index, name in enumerate(names)],
            'unlinked_doses': [dose for index, dose in enumerate(doses) if index not in linked],
            'dose_linking': LINKING_RULE}


def synthetic_negatives():
    return [dict(row, dose_label_protocol=PROTOCOL, dose_mentions=[], dose_relations=[])
            for row in name_spans.synthetic_negatives()]


def score(references, predictions, linking_rule=LINKING_RULE):
    counts = {key: [0, 0, 0] for key in ('names', 'doses', 'medication_dose_pairs')}
    def span_key(span):
        return span['start'], span['end']
    for record, prediction in zip(references, predictions, strict=True):
        reference_spans(record)
        names = name_spans.resolve_response(record, json.dumps(name_spans.reference_payload(record)))
        doses = record['dose_mentions']
        predicted_names, predicted_doses = resolve(record['text'], prediction)
        predicted_relations = link_doses(record['text'], predicted_names, predicted_doses, rule=linking_rule)
        expected = {'names': {span_key(s) for s in names}, 'doses': {span_key(s) for s in doses},
                    'medication_dose_pairs': {(span_key(names[r['medication_index']]), span_key(doses[r['dose_index']]))
                                              for r in record['dose_relations']}}
        actual = {'names': {span_key(s) for s in predicted_names}, 'doses': {span_key(s) for s in predicted_doses},
                  'medication_dose_pairs': {(span_key(predicted_names[r['medication_index']]), span_key(predicted_doses[r['dose_index']]))
                                            for r in predicted_relations}}
        for key in counts:
            for i, value in enumerate((len(actual[key] & expected[key]), len(actual[key]-expected[key]), len(expected[key]-actual[key]))):
                counts[key][i] += value
    result = {'examples': len(references)}
    for key, (tp, fp, fn) in counts.items():
        result[key] = {'tp': tp, 'fp': fp, 'fn': fn, 'precision': tp/(tp+fp) if tp+fp else None,
                       'recall': tp/(tp+fn) if tp+fn else None, 'f1': 2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else None}
    return result
