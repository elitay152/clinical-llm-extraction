"""Frozen contextual span features for medication–attribute relations."""
import hashlib
import json
import math
import re

from scripts import medication_doses as doses, medication_regimens as regimens, name_spans

VERSION = 'frozen-span-relations-v1'
FIELDS = ('dose', 'route', 'frequency')
LAYOUT = ('rule', 'before', 'distance', 'near10', 'near40', 'near120',
          'intervening_names', 'sentences', 'semicolons', 'newlines', 'commas',
          'coordination', 'parenthesis', 'colon', 'same_name', 'nearest')


def fingerprint(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def entities(text, payload):
    names, amounts = doses.resolve(text, payload)
    return names, dict(dose=amounts, **{f: regimens.attribute_spans(text, payload, f) for f in FIELDS[1:]})


def reference_payload(record, candidates=False):
    regimens.reference_spans(record)
    tokens = name_spans.source_tokens(record['text'])
    starts = {t['start']: i + 1 for i, t in enumerate(tokens)}
    ends = {t['end']: i + 1 for i, t in enumerate(tokens)}
    payload = name_spans.reference_payload(record)
    names = name_spans.resolve_response(record, json.dumps(payload))
    options = regimens.candidates(record) if candidates else {}
    for field in FIELDS:
        spans = record[f'{field}_mentions']
        if candidates:
            spans = doses.dose_candidates(record['text']) if field == 'dose' else options[field]
            spans = [s for s in spans if not any(s['start'] < n['end'] and n['start'] < s['end'] for n in names)]
        payload[f'{field}_spans'] = [[starts[s['start']], ends[s['end']]] for s in spans]
    return payload


def rule_links(text, payload):
    names, attrs = entities(text, payload)
    return dict(dose=doses.link_doses(text, names, attrs['dose']), **{
        f: regimens.link_attributes(text, names, attrs['dose'], attrs['route'], attrs['frequency'], f)
        for f in FIELDS[1:]})


def pairs(text, payload, limit=4096):
    names, attrs = entities(text, payload)
    if len(names) * sum(map(len, attrs.values())) > limit:
        raise ValueError('Relation candidate limit exceeded')
    rules = rule_links(text, payload)
    rows = []
    for kind, field in enumerate(FIELDS):
        accepted = {(r['medication_index'], r[f'{field}_index']) for r in rules[field]}
        for a, attr in enumerate(attrs[field]):
            distances = [max(0, attr['start'] - n['end'], n['start'] - attr['end']) for n in names]
            for m, name in enumerate(names):
                left, right = sorted((name, attr), key=lambda s: s['start'])
                gap = text[left['end']:right['start']]
                distance = distances[m]
                features = [float((m, a) in accepted), float(name['start'] < attr['start']),
                            math.log1p(distance) / 7, *(float(distance <= k) for k in (10, 40, 120)),
                            min(5, sum(left['end'] <= n['start'] < right['start'] for n in names)) / 5,
                            min(5, len(re.findall(r'[.!?]', gap))) / 5,
                            min(5, gap.count(';')) / 5, min(5, gap.count('\n')) / 5,
                            min(5, gap.count(',')) / 5, float(bool(re.search(r'\b(?:and|or)\b', gap))),
                            float('(' in gap or ')' in gap), float(':' in gap),
                            min(5, sum(n['text'].lower() == name['text'].lower() for n in names)) / 5,
                            float(distance == min(distances))]
                rows.append(dict(kind=kind, medication_index=m, attribute_index=a, layout=features))
    return rows


def targets(record, payload, rows):
    names, attrs = entities(record['text'], payload)
    key = lambda s: (s['start'], s['end'])
    gold_names = name_spans.resolve_response(record, json.dumps(name_spans.reference_payload(record)))
    positives = {}
    ambiguous = {}
    for field in FIELDS:
        links = record[f'{field}_relations']
        positives[field] = {(key(gold_names[r['medication_index']]), key(record[f'{field}_mentions'][r[f'{field}_index']])) for r in links}
        linked = {r[f'{field}_index'] for r in links}
        ambiguous[field] = {key(s) for i, s in enumerate(record[f'{field}_mentions']) if i not in linked}
        candidates = {key(s) for s in attrs[field]}
        if any(a not in candidates for _, a in positives[field]):
            raise ValueError('Source candidates omit a positive reference attribute')
    result = []
    for row in rows:
        field = FIELDS[row['kind']]
        n, a = key(names[row['medication_index']]), key(attrs[field][row['attribute_index']])
        result.append(None if a in ambiguous[field] else int((n, a) in positives[field]))
    return result


def features(extractor, text, payload):
    import torch
    encoded, word_ids = extractor.classifier.encode_record(
        {'text': text}, extractor.tokenizer, extractor.config['max_length'], labeled=False)
    with torch.no_grad(), torch.autocast('cpu', dtype=torch.bfloat16,
                                        enabled=extractor.config['compute_dtype'] == 'bfloat16'):
        hidden = extractor.model(**{k: torch.tensor([v]) for k, v in encoded.items()}, output_hidden_states=True).hidden_states[-1][0].float()
    tokens = name_spans.source_tokens(text)
    names, attrs = entities(text, payload)
    def pool(span):
        words = {i for i, t in enumerate(tokens) if span['start'] <= t['start'] and t['end'] <= span['end']}
        indices = [i for i, word in enumerate(word_ids) if word in words]
        if not indices:
            raise ValueError('Entity has no encoder representation')
        return hidden[indices].mean(0)
    name_vectors = [pool(n) for n in names]
    attr_vectors = {f: [pool(s) for s in attrs[f]] for f in FIELDS}
    rows = pairs(text, payload)
    vectors = []
    for row in rows:
        n = name_vectors[row['medication_index']]
        a = attr_vectors[FIELDS[row['kind']]][row['attribute_index']]
        vectors.append(torch.cat((n, a, n * a)))
    width = hidden.shape[-1] * 3
    return rows, (torch.stack(vectors) if vectors else torch.empty((0, width))), torch.tensor([r['layout'] for r in rows]).reshape(-1, len(LAYOUT))


def make_model(width, config):
    import torch
    class RelationModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.semantic = torch.nn.Sequential(torch.nn.LayerNorm(width), torch.nn.Linear(width, config['semantic_width']), torch.nn.GELU())
            self.layout = torch.nn.Sequential(torch.nn.Linear(len(LAYOUT), config['layout_width']), torch.nn.GELU())
            self.head = torch.nn.Sequential(torch.nn.Dropout(config['dropout']), torch.nn.Linear(config['semantic_width'] + config['layout_width'], 3))
            torch.nn.init.zeros_(self.head[-1].weight)
            torch.nn.init.zeros_(self.head[-1].bias)
        def forward(self, semantic, layout, kinds):
            residual = self.head(torch.cat((self.semantic(semantic), self.layout(layout)), -1))
            return residual.gather(1, kinds[:, None]).squeeze(1) + (layout[:, 0] * 2 - 1) * config['rule_prior_logit']
    return RelationModel()


def decode(rows, probabilities, threshold=0.5):
    if not math.isfinite(threshold) or not 0 < threshold < 1:
        raise ValueError('Invalid relation decision threshold')
    links = {f: [] for f in FIELDS}
    if len(rows) != len(probabilities):
        raise ValueError('Relation probability count differs from candidates')
    for row, probability in zip(rows, probabilities, strict=True):
        if not math.isfinite(probability) or not 0 <= probability <= 1:
            raise ValueError('Invalid relation probability')
        if probability >= threshold:
            field = FIELDS[row['kind']]
            links[field].append({'medication_index': row['medication_index'], f'{field}_index': row['attribute_index']})
    return links
