"""Align medication-name references to BIO labels and decode source spans."""
import math

from scripts import name_spans

LABELS = ['O', 'B-MED', 'I-MED']
VERSION = 'medication-name-bio-v1'


def word_labels(record):
    labels = [0] * len(name_spans.source_tokens(record['text']))
    for start, end in name_spans.reference_payload(record)['medication_name_spans']:
        if any(labels[start-1:end]):
            raise ValueError('Overlapping reference medication spans are unsupported')
        labels[start-1] = 1
        labels[start:end] = [2] * (end-start)
    return labels


def encode_record(record, tokenizer, max_length=512, labeled=True):
    words = name_spans.source_tokens(record['text'])
    if not words:
        raise ValueError('Empty source has no classifiable tokens')
    encoded = tokenizer([word['text'] for word in words], is_split_into_words=True, truncation=False)
    if len(encoded['input_ids']) > max_length:
        raise ValueError('Source exceeds classifier context; refusing truncation')
    word_ids = encoded.word_ids()
    if {i for i in word_ids if i is not None} != set(range(len(words))):
        raise ValueError('Tokenizer omitted a source token')
    result = dict(encoded)
    if labeled:
        labels = word_labels(record)
        previous = None
        result['labels'] = []
        for word in word_ids:
            result['labels'].append(labels[word] if word is not None and word != previous else -100)
            previous = word
    return result, word_ids


def decode_predictions(word_ids, predictions, word_count):
    if len(word_ids) != len(predictions):
        raise ValueError('Prediction length differs from encoded tokens')
    labels = {}
    for word, prediction in zip(word_ids, predictions):
        if word is not None and word not in labels:
            if type(prediction) is not int or prediction not in range(len(LABELS)):
                raise ValueError('Unknown BIO label')
            labels[word] = prediction
    if set(labels) != set(range(word_count)):
        raise ValueError('Predictions do not cover every source token')
    pairs = []
    start = None
    for index in range(word_count):
        label = labels[index]
        if start is not None and label != 2:
            pairs.append([start+1, index])
            start = None
        # An orphan I starts a span; it cannot extend a preceding O.
        if label == 1 or (label == 2 and start is None):
            start = index
    if start is not None:
        pairs.append([start+1, word_count])
    return {'medication_name_spans': pairs}


def loss_weights(counts, rule='sqrt_inverse_frequency'):
    if any(counts.get(label, 0) <= 0 for label in range(3)):
        raise ValueError('Training needs examples of all three BIO labels')
    if rule == 'uniform':
        return [1.0] * 3
    if rule != 'sqrt_inverse_frequency':
        raise ValueError('Unsupported class weighting rule')
    raw = [math.sqrt(sum(counts.values())/(3*counts[label])) for label in range(3)]
    return [weight/(sum(raw)/3) for weight in raw]
