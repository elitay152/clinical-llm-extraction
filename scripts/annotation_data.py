"""Validation shared by annotation preparation and exported-workspace checks."""
import argparse
import json
from pathlib import Path

FORMAT = 'clinical-medication-pilot-v1'
STATUSES = {'current', 'discontinued', 'historical', 'hypothetical', 'unknown'}
STAGES = {'unannotated', 'draft', 'reviewed', 'adjudicated'}
FIELDS = {'name', 'strength', 'dose', 'route', 'frequency', 'status', 'evidence'}


def validate_document(doc):
    if not isinstance(doc, dict) or set(doc) != {'id', 'text', 'medications', 'annotation_status'}:
        raise ValueError('Document must contain exactly id, text, medications, annotation_status')
    if not isinstance(doc['id'], str) or not doc['id'].strip() or not isinstance(doc['text'], str):
        raise ValueError('Invalid document ID or text')
    if doc['annotation_status'] not in STAGES or not isinstance(doc['medications'], list):
        raise ValueError('Invalid annotation stage or medication list')
    for med in doc['medications']:
        if not isinstance(med, dict) or set(med) != FIELDS:
            raise ValueError('Medication fields do not match schema')
        if not isinstance(med['name'], str) or not med['name'].strip() or med['status'] not in STATUSES:
            raise ValueError('Medication needs a name and valid status')
        for field in ('strength', 'dose', 'route', 'frequency'):
            if med[field] is not None and (not isinstance(med[field], str) or not med[field].strip()):
                raise ValueError(f'{field} must be null or a nonempty string')
        if not isinstance(med['evidence'], list) or not med['evidence']:
            raise ValueError('Every medication requires supporting evidence')
        for span in med['evidence']:
            if not isinstance(span, dict) or set(span) != {'start', 'end', 'text'}:
                raise ValueError('Evidence needs start, end, text')
            start, end = span['start'], span['end']
            if type(start) is not int or type(end) is not int or not 0 <= start < end <= len(doc['text']):
                raise ValueError('Evidence offsets are outside source text')
            if doc['text'][start:end] != span['text']:
                raise ValueError('Evidence text does not match source offsets')
        if not any(med['name'] in span['text'] for span in med['evidence']):
            raise ValueError('Evidence must include the medication name as written')
    return doc


def validate_workspace(workspace, reference=None):
    if not isinstance(workspace, dict) or workspace.get('format') != FORMAT:
        raise ValueError('Unsupported workspace format')
    if workspace.get('role') != 'development_pilot' or workspace.get('protocol_version') != '1':
        raise ValueError('This tool only accepts development pilot protocol v1')
    documents = workspace.get('documents')
    if not isinstance(documents, list) or not documents:
        raise ValueError('Workspace has no documents')
    ids = set()
    expected = {d['id']: d['text'] for d in reference['documents']} if reference else None
    metadata = workspace.get('metadata')
    if not isinstance(metadata, dict) or not isinstance(workspace.get('provenance'), dict):
        raise ValueError('Workspace metadata and provenance are required')
    for doc in documents:
        validate_document(doc)
        if doc['id'] in ids:
            raise ValueError('Duplicate document ID')
        ids.add(doc['id'])
        if expected is not None and expected.get(doc['id']) != doc['text']:
            raise ValueError('Source text or document membership changed')
        if doc['id'] not in workspace['provenance']:
            raise ValueError('Missing source provenance')
        if reference and workspace['provenance'][doc['id']] != reference['provenance'][doc['id']]:
            raise ValueError('Source provenance changed')
        meta = metadata.get(doc['id'], {})
        if not isinstance(meta, dict):
            raise ValueError('Invalid annotation metadata')
        if doc['annotation_status'] != 'unannotated':
            if not isinstance(meta.get('annotator'), str) or not meta['annotator'].strip():
                raise ValueError('Saved annotations require an annotator alias')
        if doc['annotation_status'] in {'reviewed', 'adjudicated'}:
            if meta.get('review_confirmed') is not True:
                raise ValueError('Reviewed documents require explicit review confirmation')
            if not doc['medications'] and meta.get('negative_confirmed') is not True:
                raise ValueError('Reviewed empty lists require explicit negative confirmation')
        if doc['annotation_status'] == 'adjudicated':
            raise ValueError('Independent adjudication is not supported by this first-pass tool')
    if expected is not None and ids != set(expected):
        raise ValueError('Document membership changed')
    return workspace


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('workspace', type=Path)
    parser.add_argument('--reference', type=Path, default=Path(__file__).resolve().parents[1] / 'data/annotation/pilot-v1.json')
    args = parser.parse_args()
    workspace = json.loads(args.workspace.read_text())
    reference = json.loads(args.reference.read_text())
    validate_workspace(workspace, reference)
    counts = {stage: sum(d['annotation_status'] == stage for d in workspace['documents']) for stage in sorted(STAGES)}
    print(json.dumps({'valid': True, 'stages': counts, 'independent_gold': False}, indent=2))


if __name__ == '__main__':
    main()
