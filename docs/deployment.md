# Hosted demo

Create a blank **Gradio** Space in Hugging Face using the hardware available
on your account. The hosted app requests GPU time only for extraction on
ZeroGPU, and otherwise falls back to CPU. Its runtime uses PyTorch 2.8.0. A public Space lets visitors try the
app, while its checkpoint stays in a separate private model repository.

Add a fine-grained **deployment token** as `HF_TOKEN` in the cloud development
environment. It needs write permission to the target Space and private model
repository. If you create those repositories yourself, scope the token to them.
Then deploy:

```bash
.venv/bin/python scripts/deploy_demo.py artifacts/my-classifier \
  --output artifacts/demo-deployment --space YOUR_USERNAME/clinical-medication-demo \
  --model-repo YOUR_USERNAME/clinical-medication-checkpoint
```

In the Space's Settings, add a separate `HF_TOKEN` **secret** with read permission
only to that private model repository. Do not put the deployment token in the
Space. No teacher API key is needed. The uploader sets `MODEL_REPO` and the exact
`MODEL_REVISION` as Space variables. The Space downloads that checkpoint at first
inference and caches it in temporary storage; no storage bucket is needed.

An existing Space keeps its visibility. A newly created Space is public. The
uploader rejects public model repositories and never includes weights in the
Space upload. Only model files and minimal inference settings enter the private
model repository; training IDs, datasets, labels, predictions, evaluation reports,
and credentials are excluded from both uploads.

Omit `--space` to package locally without uploading. Use a new directory under
ignored `artifacts/` each time. The package contains separate `space/` and `model/`
directories. To test the Gradio interface locally:

```bash
.venv/bin/python -m pip install gradio==5.29.0
MODEL_PATH="$PWD/artifacts/demo-deployment/model" PYTHONPATH="$PWD" \
  .venv/bin/python artifacts/demo-deployment/space/app.py
```

Use synthetic notes. The application does not store submitted notes; hosted
traffic also passes through the hosting provider. Public Space code and the
model repository identifier are visible, while the checkpoint and token remain
private. Weights are not added to the public GitHub repository.
