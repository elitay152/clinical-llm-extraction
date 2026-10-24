"""Gradio interface for source-aligned medication-name extraction."""
from functools import lru_cache
import os
import json
from pathlib import Path

import gradio as gr
import spaces
import torch
from huggingface_hub import snapshot_download

from scripts.predict_medication_names import MedicationNameExtractor
from scripts import medication_attributes


@lru_cache(maxsize=3)
def extractor(task='Medication names'):
    rule_mode=task=='Names, doses, routes, frequencies (experimental)' and os.environ.get('REGIMEN_RULES_ENABLED')=='1'
    path = os.environ.get('MODEL_PATH')
    if not path:
        folder=('regimen' if task=='Names, doses, routes, frequencies (experimental)' and not rule_mode else
                'dose' if task!='Medication names' else '')
        prefix=folder+'/' if folder else ''
        revision=os.environ[f'{folder.upper()}_MODEL_REVISION'] if folder else os.environ['MODEL_REVISION']
        path = snapshot_download(repo_id=os.environ['MODEL_REPO'], revision=revision,
                                 token=os.environ.get('HF_TOKEN'),
                                 allow_patterns=[prefix+name for name in ('config.json','run.json','tokenizer.json',
                                                 'tokenizer_config.json','special_tokens_map.json','vocab.txt','model.safetensors')])
        if folder:
            path=os.path.join(path,folder)
    device='cuda' if torch.cuda.is_available() else 'cpu'
    report=json.loads((Path(path)/'run.json').read_text())
    if report['alignment_version']==medication_attributes.VERSION:
        base=snapshot_download(repo_id=os.environ['MODEL_REPO'],revision=os.environ['DOSE_MODEL_REVISION'],
                               token=os.environ.get('HF_TOKEN'),allow_patterns=['dose/*'])
        return MedicationNameExtractor(Path(base)/'dose',device=device,attribute_run=path)
    return MedicationNameExtractor(path,device=device,attribute_candidates=rule_mode)


@spaces.GPU(duration=60)
def extract(text, task='Medication names'):
    if task not in TASKS:
        raise gr.Error('Unknown extraction task')
    if not isinstance(text, str) or not text.strip():
        raise gr.Error('Provide nonempty note text')
    if len(text.encode('utf-8')) > 65536:
        raise gr.Error('Note must contain at most 64 KiB')
    try:
        result = extractor(task).predict(text)
    except ValueError as error:
        raise gr.Error(str(error)) from error
    segments = []
    cursor = 0
    highlighted=[(span,'Medication') for span in result['medication_names']]
    for field,label in [('dose','Dose'),('route','Route'),('frequency','Frequency')]:
        highlighted.extend((span,label) for span in result.get(f'{field}_spans',[]))
    for span,kind in sorted(highlighted,key=lambda item:item[0]['start']):
        segments.append((text[cursor:span['start']], None))
        segments.append((span['text'], kind))
        cursor = span['end']
    segments.append((text[cursor:], None))
    return segments, result


TASKS=['Medication names']
if os.environ.get('DOSE_MODEL_REVISION'):
    TASKS.append('Names and doses (experimental)')
if os.environ.get('REGIMEN_MODEL_REVISION') or os.environ.get('REGIMEN_RULES_ENABLED')=='1':
    TASKS.append('Names, doses, routes, frequencies (experimental)')


with gr.Blocks(title='Medication extractor', analytics_enabled=False) as demo:
    gr.Markdown('# Medication extractor\n'
                'Extract medication names with their source evidence. '
                + ('Choose the experimental dose mode to include amounts. ' if len(TASKS)>1 else '')
                + ('The regimen mode also extracts routes and frequencies. ' if 'routes, frequencies' in TASKS[-1] else 'Route and frequency are outside this task. ')
                + 'Medication status is not predicted. '
                'Teacher-reference agreement does not establish clinical accuracy.')
    task=gr.Dropdown(TASKS,value=TASKS[0],label='Extraction task',visible=len(TASKS)>1)
    note = gr.Textbox(label='Synthetic note', lines=5, max_lines=12,
                      value='The patient takes aspirin 81 mg daily and metformin 500 mg twice daily.')
    gr.Examples(examples=[['Take aspirin 81 mg orally once daily. Take metformin 500 mg orally twice daily.'],
                          ['Warfarin was discontinued. The patient now takes apixaban 5 mg twice daily.'],
                          ['Laboratory results: sodium 138 mmol/L, potassium 4.2 mmol/L, glucose 95 mg/dL.']],
                inputs=note)
    button = gr.Button('Extract', variant='primary')
    evidence = gr.HighlightedText(label='Source evidence', combine_adjacent=False,
                                  color_map={'Medication':'#bcebd9','Dose':'#f8dfab','Route':'#bddcf4','Frequency':'#e3cff5'})
    with gr.Accordion('JSON output',open=False):
        output = gr.JSON(label='Structured extraction')
    button.click(extract, inputs=[note,task], outputs=[evidence, output], concurrency_limit=1)
    gr.Markdown('Use synthetic text. The application does not save submitted notes or call a model API. '
                'Inputs beyond the checkpoint context limit are rejected.')


if __name__ == '__main__':
    demo.queue(max_size=8).launch(server_name='0.0.0.0', server_port=int(os.environ.get('PORT', '7860')),
                                 show_error=False, show_api=False)
