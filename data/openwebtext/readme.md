
## openwebtext dataset

after running `prepare.py` (preprocess) we get:

- train.bin is ~17GB, val.bin ~8.5MB
- train has ~9B tokens (9,035,582,489)
- val has ~4M tokens (4,434,606)

this came from 8,013,769 documents in total.

The preparation script pins the Hugging Face dataset
`Skylion007/openwebtext` at revision
`79d93d786212f7344586290adb811d4ae6a1762c`. The Slurm preparation job checks
this document count and the expected train and validation token counts before
recording dataset fingerprints. These pinned modern-pipeline split counts move
291 tokens from validation to training relative to nanoGPT's historical split,
while preserving its exact 9,040,017,095-token corpus total.

references:

- OpenAI's WebText dataset is discussed in [GPT-2 paper](https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)
- [OpenWebText](https://skylion007.github.io/OpenWebTextCorpus/) dataset
