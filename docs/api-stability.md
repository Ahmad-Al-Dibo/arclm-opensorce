# API Stability

ArcLM `0.9.0` uses a central manifest in `arclm.stability`.

Stable APIs are covered by `tests/fixtures/api_snapshot_0_9.json`. Changes to
that list should be treated as compatibility changes and reviewed explicitly.

Stability labels:

| Label | Meaning |
| --- | --- |
| `stable` | Intended to remain compatible during the `0.9.x` line. |
| `provisional` | Public but still being shaped before `1.0`. |
| `experimental` | Useful but may change with shorter notice. |
| `deprecated` | Functional compatibility path with a replacement. |
| `internal` | Not part of the supported public API. |

Use:

```python
from arclm.stability import api_manifest, stable_api_paths

print(stable_api_paths())
```
