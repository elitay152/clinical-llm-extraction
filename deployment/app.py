"""Gradio interface for source-aligned medication-name extraction."""
from functools import lru_cache
import os

import gradio as gr
from huggingface_hub import snapshot_download

from scripts.predict_medication_names import MedicationNameExtractor


@lru_cache(maxsize=1)
def extractor():
    path = os.environ.get('MODEL_PATH')
    if not path:
        path = snapshot_download(repo_id=os.environ['MODEL_REPO'],
                                 revision=os.environ['MODEL_REVISION'],
                                 token=os.environ.get('HF_TOKEN'),
                                 allow_patterns=['*.json', '*.txt', 'model.safetensors'])
    return MedicationNameExtractor(path)


def extract(text):
    try:
        result = extractor().predict(text)
    except ValueError as error:
        raise gr.Error(str(error)) from error
    segments = []
    cursor = 0
    for span in result['medication_names']:
        segments.append((text[cursor:span['start']], None))
        segments.append((span['text'], 'Medication'))
        cursor = span['end']
    segments.append((text[cursor:], None))
    return segments, result


with gr.Blocks(title='Medication name extractor', analytics_enabled=False) as demo:
    gr.Markdown('# Medication name extractor\n'
                'Extract medication names with their source evidence. Names only; '
                'dose, route, frequency, and status are outside this task. '
                'Teacher-reference agreement does not establish clinical accuracy.')
    note = gr.Textbox(label='Synthetic note', lines=5, max_lines=12,
                      value='The patient takes aspirin 81 mg daily and metformin 500 mg twice daily.')
    gr.Examples(examples=[['Warfarin was discontinued. The patient now takes apixaban 5 mg twice daily.'],
                          ['Laboratory results: sodium 138 mmol/L, potassium 4.2 mmol/L, glucose 95 mg/dL.']],
                inputs=note)
    button = gr.Button('Extract names', variant='primary')
    evidence = gr.HighlightedText(label='Source evidence', combine_adjacent=False,
                                  color_map={'Medication': '#bcebd9'})
    output = gr.JSON(label='JSON output')
    button.click(extract, inputs=note, outputs=[evidence, output], concurrency_limit=1)
    gr.Markdown('Use synthetic text. The application does not save submitted notes or call a model API. '
                'Inputs beyond the checkpoint context limit are rejected.')


if __name__ == '__main__':
    demo.queue(max_size=8).launch(server_name='0.0.0.0', server_port=int(os.environ.get('PORT', '7860')),
                                 show_error=False, show_api=False)
