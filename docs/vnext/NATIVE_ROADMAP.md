# ArcLM Native Framework Roadmap

Status: PARTIAL dependency inventory and migration direction.

ArcLM's rule is: ArcLM owns the logic; external libraries provide capabilities.
This roadmap classifies current dependencies by how they should evolve.

| Dependency / Usage | Current Role | Decision | Notes |
| --- | --- | --- | --- |
| PyTorch tensors/autograd/modules | Native model compute and training | ISOLATE BEHIND BACKEND | Keep as transitional Torch backend; do not rewrite tensor algebra. |
| `torch.nn` layers | Native `ArcLM` architecture | ISOLATE BEHIND BACKEND | Architecture semantics should be ArcLM-owned; implementation can use Torch modules. |
| `torch.optim` | Optimizer creation | ISOLATE BEHIND BACKEND | Trainer should own optimizer semantics, Torch supplies implementation. |
| `torch.utils.data` | Current batching | REPLACE WITH ARCLM LOGIC / BACKEND | Public `Dataset.prepare()` should own sequence/batch semantics; Torch DataLoader may remain internal. |
| `torch.cuda` checks | Device selection | REPLACE PUBLIC USAGE | Use `Runtime.auto()` publicly; CUDA probing remains internal. |
| `torch.save` / `torch.load` | Legacy checkpoints/loaders | ISOLATED / REPLACE | Removed from native `.arcmodel`; still used by legacy checkpoints and compatibility loaders. |
| Transformers `AutoModelForCausalLM.from_pretrained` | External model loading | OPTIONAL INTEGRATION | Keep only as compatibility importer while ArcLM model registry/resolver matures. |
| Transformers `AutoTokenizer.from_pretrained` | External tokenizers | OPTIONAL INTEGRATION | Use through tokenizer resolver/spec, not as public architecture. |
| Transformers `save_pretrained` | External export | OPTIONAL INTEGRATION | Belongs under `model.export(...)`, not native `model.save(...)`. |
| PEFT `get_peft_model` / `PeftModel` | LoRA integration | ISOLATE BEHIND ADAPTER BACKEND | Native LoRA should be ArcLM-owned first for ArcLM architectures; PEFT remains compatibility. |
| Safetensors | Native tensor serialization | KEEP AS LOW-LEVEL DEPENDENCY | Used for `.arcmodel` and `.arcadapter` tensor bytes. ArcLM owns manifest/layout/integrity. |
| SentencePiece | Tokenization implementation | KEEP AS LOW-LEVEL DEPENDENCY | ArcLM owns tokenizer workflow/spec; SentencePiece can provide subword capability. |
| NumPy | Metrics/data helpers | KEEP AS LOW-LEVEL DEPENDENCY | Acceptable utility dependency where useful. |
| Flask/FastAPI | Optional local interfaces | OPTIONAL INTEGRATION | Not part of core engine. |
| BeautifulSoup/PyYAML/tqdm | Preprocess/support tooling | OPTIONAL INTEGRATION | Keep outside core imports. |

## Native Workstreams

1. Runtime backend boundary: formalize Torch backend object behind `Runtime`.
2. Dataset engine: move sequence batching out of public `torch.utils.data`
   concepts.
3. Artifact engine: `.arcmodel` and `.arcadapter` are native; `.arcckpt`
   remains planned.
4. Model engine: keep native ArcLM architecture first, add registry-resolved
   specs before importing external families.
5. Training engine: route `train_model`, native SFT and Lab training through the
   same `Trainer` core.
6. External compatibility: demote Transformers/PEFT to import/export adapters.
