# Model loading en support

## Supportniveaus zijn productlogica

Model support is niet alleen "Transformers kan het laden". ArcLM gebruikt vier
niveaus:

- `official`
- `experimental`
- `compatible_unverified`
- `unsupported`

Deze waarden staan in `supported_models.py` en worden runtime gebruikt door
`models.inspect_model_support`.

## Huidige waarheid

- ArcLM native checkpoint: official voor compacte native causal LM workflows.
- GPT-2 via Hugging Face: official, met tiny GPT-2 als geautomatiseerd
  certificatie-artifact.
- Llama via Hugging Face: experimental, tiny random Llama CPU-certificatie
  bestaat, maar echte Llama checkpoints zijn niet gecertificeerd.
- Qwen: experimental op basis van voorbeelden, niet core CI-certificatie.
- Mistral/Gemma/Falcon: compatible_unverified, loader kan proberen maar ArcLM
  claimt geen familiecertificatie.
- Encoder-only/seq2seq: unsupported.

## Welke loader gebruik je?

Voor nieuwe code: `arclm.models.inspect_model_support` en
`arclm.models.load_model`.

Vermijd nieuwe afhankelijkheid op:

- `external_inference.load_any_model` voor nieuwe publieke flows
- `training.unified.PreTrainedModelLoader` voor echte HF support
- directe `inference.load_model` behalve wanneer je bewust alleen native
  `.pth` ArcLM checkpoints ondersteunt

## Verborgen veiligheidsverschil

`ExternalModelConfig.trust_remote_code` default staat historisch op `True`.
De nieuwe facade gebruikt `trust_remote_code=False`. Dit verschil is belangrijk:
nieuwe features moeten de nieuwe facade volgen en remote code nooit stil
aanzetten.

## Runtime-inspectie

`inspect_model_support` gebruikt:

1. `inspect_model_source` om bronsoort te bepalen.
2. `AutoConfig.from_pretrained` om architectuur/model_type te inspecteren.
3. `AutoTokenizer.from_pretrained` om tokenizer beschikbaarheid te controleren.
4. heuristieken zoals `CausalLM`, `LMHeadModel`, `GPT2LMHeadModel`.
5. metadata lookup per familie.

Het laadt niet altijd volledige gewichten. Dat is bewust om support checks
goedkoop en veiliger te houden.

## Certificatieprotocol

`certify_model_family` test config/tokenizer, model load, deterministische
greedy generation, volgorde van meerdere prompts, evaluatie, optioneel een
minimale SFT stap, save/reload en inference na reload.

Nieuwe "official" model support hoort pas na een vergelijkbare test in CI of
minstens een reproduceerbaar klein testmodel.
