

def print_model_summary(
    model,
    architecture,
    config,
    runtime,
    ) -> None:
    """Print a model summary."""

    print(f"Model: {model.__class__.__name__}")
    print(f"Architecture: {architecture.architecture_id}")
    print(f"parameters: {sum(p.numel() for p in model.parameters())}")
    print(f"Config: {config.to_dict()}")
    print(f"Runtime: {runtime.to_dict()}")
