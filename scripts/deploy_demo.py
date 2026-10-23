"""Package a Gradio demo and deploy it with a private checkpoint."""
import argparse
import json
import os
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

SCRIPTS = ('predict_medication_names.py', 'token_classifier.py', 'name_spans.py')
MODEL_FILES = ('config.json', 'model.safetensors', 'tokenizer.json',
               'tokenizer_config.json', 'special_tokens_map.json', 'vocab.txt')


def package(run, output):
    run, output = Path(run), Path(output)
    report = json.loads((run/'run.json').read_text())
    # Deployment metadata excludes training IDs, label hashes, and workspace paths.
    metadata = {key: report[key] for key in ('alignment_version', 'source_token_version', 'threads')}
    metadata['config'] = {key: report['config'][key] for key in ('max_length', 'compute_dtype')}
    for name in MODEL_FILES:
        if not (run/name).is_file():
            raise ValueError(f'Missing checkpoint file: {name}')
    output.mkdir(parents=True, exist_ok=False)
    (output/'model').mkdir()
    (output/'space/scripts').mkdir(parents=True)
    for name in MODEL_FILES:
        shutil.copyfile(run/name, output/'model'/name)
    (output/'model/run.json').write_text(json.dumps(metadata, indent=2)+'\n')
    for name in SCRIPTS:
        shutil.copyfile(ROOT/'scripts'/name, output/'space/scripts'/name)
    for source, target in [('app.py', 'app.py'), ('requirements.txt', 'requirements.txt'),
                           ('space.md', 'README.md')]:
        shutil.copyfile(ROOT/'deployment'/source, output/'space'/target)


def upload(output, space, model_repo, token):
    from huggingface_hub import HfApi
    api = HfApi(token=token)
    api.create_repo(model_repo, repo_type='model', private=True, exist_ok=True)
    if not api.repo_info(model_repo, repo_type='model').private:
        raise ValueError('Checkpoint repository must be private')
    commit = api.upload_folder(repo_id=model_repo, repo_type='model', folder_path=str(output/'model'),
                               commit_message='Upload inference checkpoint')
    from huggingface_hub.errors import RepositoryNotFoundError
    try:
        info = api.space_info(space)
    except RepositoryNotFoundError:
        api.create_repo(space, repo_type='space', space_sdk='gradio', private=False)
    else:
        if info.sdk != 'gradio':
            raise ValueError('Target Space must use the Gradio SDK')
    api.add_space_variable(space, key='MODEL_REPO', value=model_repo)
    api.add_space_variable(space, key='MODEL_REVISION', value=commit.oid)
    api.upload_folder(repo_id=space, repo_type='space', folder_path=str(output/'space'),
                      commit_message='Deploy medication-name demo')
    return f'https://huggingface.co/spaces/{space}'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    parser.add_argument('--output', type=Path, required=True, help='New staging directory under ignored artifacts/')
    parser.add_argument('--space', help='Deploy to OWNER/SPACE; otherwise package only')
    parser.add_argument('--model-repo', help='Private OWNER/MODEL checkpoint repository')
    args = parser.parse_args()
    if args.space and not args.model_repo:
        parser.error('--model-repo is required with --space')
    if not args.output.resolve().is_relative_to((ROOT/'artifacts').resolve()):
        parser.error('Staging output must be under artifacts/ to keep weights outside Git')
    token = os.environ.get('HF_TOKEN')
    if args.space and not token:
        parser.error('Set HF_TOKEN securely in environment settings to deploy')
    package(args.run, args.output)
    print(upload(args.output, args.space, args.model_repo, token) if args.space else f'Packaged: {args.output}')


if __name__ == '__main__':
    main()
