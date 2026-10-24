import json
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import Mock, patch

from scripts.deploy_demo import MODEL_FILES, package, upload


class DeploymentTests(unittest.TestCase):
    def test_package_excludes_training_metadata_and_unrelated_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run = root/'run'
            run.mkdir()
            for name in MODEL_FILES:
                (run/name).write_text('checkpoint fixture')
            (run/'private-labels.json').write_text('private data')
            (run/'run.json').write_text(json.dumps({
                'alignment_version': 'bio', 'source_token_version': 'tokens', 'threads': 2,
                'training_ids': ['private-id'], 'silver_directory': '/private/data',
                'config': {'max_length': 512, 'compute_dtype': 'bfloat16', 'seed': 42}}))
            output = root/'package'
            package(run, output)
            metadata = json.loads((output/'model/run.json').read_text())
            self.assertEqual(set(metadata), {'alignment_version', 'source_token_version', 'threads', 'config'})
            self.assertEqual(set(metadata['config']), {'max_length', 'compute_dtype'})
            self.assertFalse((output/'model/private-labels.json').exists())
            self.assertFalse(any(path.suffix == '.safetensors' for path in (output/'space').rglob('*')))
            self.assertTrue((output/'space/app.py').is_file())
            self.assertTrue((output/'space/README.md').is_file())
            with self.assertRaises(FileExistsError):
                package(run, output)

    def test_existing_public_model_repository_is_not_uploaded(self):
        api = Mock()
        api.repo_info.return_value.private = False
        module = types.SimpleNamespace(HfApi=Mock(return_value=api))
        with patch.dict('sys.modules', {'huggingface_hub': module}):
            with self.assertRaisesRegex(ValueError, 'must be private'):
                upload(Path('/package'), 'owner/demo', 'owner/checkpoint', 'test-token')
        api.upload_folder.assert_not_called()
        self.assertTrue(api.create_repo.call_args.kwargs['private'])

    def test_existing_space_receives_code_and_private_model_revision(self):
        api = Mock()
        api.repo_info.return_value.private = True
        api.space_info.return_value.sdk = 'gradio'
        api.upload_folder.return_value.oid = 'checkpoint-commit'
        module = types.SimpleNamespace(HfApi=Mock(return_value=api))
        errors = types.SimpleNamespace(RepositoryNotFoundError=type('MissingRepo', (Exception,), {}))
        with patch.dict('sys.modules', {'huggingface_hub': module, 'huggingface_hub.errors': errors}):
            url = upload(Path('/package'), 'owner/demo', 'owner/checkpoint', 'test-token')
        self.assertEqual(url, 'https://huggingface.co/spaces/owner/demo')
        api.create_repo.assert_called_once_with('owner/checkpoint', repo_type='model',
                                                private=True, exist_ok=True)
        self.assertEqual([call.kwargs['folder_path'] for call in api.upload_folder.call_args_list],
                         ['/package/model', '/package/space'])
        api.add_space_variable.assert_any_call('owner/demo', key='MODEL_REVISION', value='checkpoint-commit')

    def test_optional_checkpoint_weights_are_only_in_private_package(self):
        from scripts import medication_doses,medication_regimens
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            report={'alignment_version':'bio','source_token_version':'tokens','threads':2,
                    'training_ids':['private-id'],'config':{'max_length':512,'compute_dtype':'bfloat16'}}
            for folder in ('names','doses','regimens'):
                run=root/folder;run.mkdir()
                for name in MODEL_FILES:
                    (run/name).write_text('checkpoint fixture')
                value=dict(report)
                if folder=='doses':value['alignment_version']=medication_doses.VERSION
                if folder=='regimens':value['alignment_version']=medication_regimens.VERSION
                (run/'run.json').write_text(json.dumps(value))
            package(root/'names',root/'output',root/'doses',root/'regimens')
            self.assertTrue((root/'output/model/dose/model.safetensors').exists())
            self.assertTrue((root/'output/model/regimen/model.safetensors').exists())
            self.assertFalse(list((root/'output/space').rglob('*.safetensors')))
            metadata=json.loads((root/'output/model/dose/run.json').read_text())
            self.assertNotIn('training_ids',metadata)
            regimen=json.loads((root/'output/model/regimen/run.json').read_text())
            self.assertNotIn('training_ids',regimen)

    def test_regimen_rules_require_a_dose_checkpoint(self):
        with self.assertRaisesRegex(ValueError,'require --dose-run'):
            package(Path('/names'),Path('/package'),regimen_rules=True)
