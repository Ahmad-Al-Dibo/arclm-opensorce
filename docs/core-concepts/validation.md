# Validation

Validation in `0.9.0` includes structured dataset reports:

- `validate_records(records, schema=..., strict=...)` validates without removing rows.
- `PreprocessPipeline` returns drop reason counts.
- `DataPipeline.validate(schema=...)` records validation failures in pipeline reports.
- Native checkpoint loading checks tokenizer compatibility in fine-tuning/continue paths.
