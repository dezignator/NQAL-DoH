
---

### File 73: `docs/api_reference.md`

```markdown
# API Reference

## Core classes

### `models.nqal_pipeline.NQALPipeline`

End-to-end NQAL pipeline.

```python
from models.nqal_pipeline import NQALPipeline, NQALConfig

pipe = NQALPipeline(NQALConfig(hidden_size=128, use_dsgat=True))
