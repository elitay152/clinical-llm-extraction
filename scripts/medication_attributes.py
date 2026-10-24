"""Train route/frequency BIO labels independently of medication names and doses."""
from scripts import medication_doses, medication_regimens, token_classifier

VERSION = 'medication-attributes-bio-v1'
LABELS = ['O','B-ROUTE','I-ROUTE','B-FREQUENCY','I-FREQUENCY']


def word_labels(record):
    return [label-4 if label>=5 else 0 for label in medication_regimens.word_labels(record)]


def encode_record(record,tokenizer,max_length=512,labeled=True):
    return token_classifier.encode_record(record,tokenizer,max_length,labeled,label_function=word_labels)


def decode_predictions(word_ids,predictions,word_count):
    spans=medication_doses.decode_predictions(word_ids,predictions,word_count)
    return {'route_spans':spans['medication_name_spans'],'frequency_spans':spans['dose_spans']}


def synthetic_negatives():
    return medication_regimens.synthetic_negatives()


def merge_payload(base,attributes):
    if set(base)!={'medication_name_spans','dose_spans'} or set(attributes)!={'route_spans','frequency_spans'}:
        raise ValueError('Attribute output cannot replace name/dose predictions')
    return dict(base,**attributes)
