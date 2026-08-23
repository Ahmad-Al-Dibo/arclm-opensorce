# ArcLM vNext Architecture

Status: PARTIAL.

The current vNext implementation starts with a narrow transitional core:

```text
arclm.vnext.Model
  -> arclm.vnext.ModelRegistry / ModelSpec
  -> arclm.runtime.Runtime
  -> arclm.artifacts.ArcModelArtifact
  -> existing arclm.model.ArcLM and arclm.tokenizer.Tokenizer
```

`arclm.vnext` is temporary. The target architecture wants `arclm.api` and
`arclm.registry` packages, but this repository already has `arclm/api.py` and
`arclm/registry.py` modules. Those modules need a compatibility migration before
package directories with the same names can exist.

## Current Principles Implemented

- ArcLM owns the public model lifecycle through `Model.create()`,
  `Model.load()`, `Model.save()`, `Model.generate()` and `Model.inspect()`.
- Runtime selection is represented as an ArcLM `Runtime`, not as a public
  `torch.device`.
- Model support is described by `ModelSpec` and capability statuses.
- `.arcmodel` is an ArcLM-owned directory artifact with a versioned manifest.
- Torch remains the internal tensor/backend implementation for the native model.

## Current Limitations

- The vNext model wrapper supports only the native ArcLM causal LM.
- `.arcmodel` currently stores weights with Torch serialization behind the
  artifact boundary. The public contract is the ArcLM manifest/container, not
  direct `torch.save`.
- Training is not yet routed through `Model` or `.arcmodel`.
- External pretrained model resolution is still handled by legacy loaders.
- No lazy/partial tensor loading yet.
