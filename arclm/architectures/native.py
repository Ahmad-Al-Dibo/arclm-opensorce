"""Built-in ArcLM-native architectures."""

from __future__ import annotations

from .base import Architecture, ArchitectureCapabilities, ArchitectureKind, CapabilitySupport, IdentityWeightMapper


class ArcLMNativeCausalLM(Architecture):
    """Compact ArcLM-native causal language-model architecture."""

    architecture_id = "arclm-native-causal-lm"
    name = "ArcLM Native Causal LM"
    version = "1"
    kind = ArchitectureKind.NATIVE
    task = "causal-lm"
    capabilities = ArchitectureCapabilities(
        load=CapabilitySupport.SUPPORTED,
        inference=CapabilitySupport.SUPPORTED,
        generation=CapabilitySupport.SUPPORTED,
        training=CapabilitySupport.SUPPORTED,
        training_from_scratch=CapabilitySupport.SUPPORTED,
        fine_tuning=CapabilitySupport.EXPERIMENTAL,
        configuration_editing=CapabilitySupport.SUPPORTED,
        layer_replacement=CapabilitySupport.PLANNED,
        custom_forward=CapabilitySupport.PLANNED,
        custom_attention=CapabilitySupport.PLANNED,
        export=CapabilitySupport.SUPPORTED,
    )
    config_schema = {
        "required": ["vocab_size"],
        "defaults": {"embed_dim": 64, "block_size": 8, "num_blocks": 2, "dropout": 0.0},
        "fields": ["vocab_size", "embed_dim", "block_size", "num_blocks", "dropout"],
    }
    component_slots = ("token_embedding", "position_embedding", "blocks", "normalization", "head")
    adapter_targets = ("blocks.0.attn.query", "blocks.0.attn.value", "head")
    runtime_requirements = {"backend": "torch", "devices": ["cpu", "cuda"]}
    weight_mapper = IdentityWeightMapper()

    def build(self, config, runtime):
        self.validate_config(config)
        from ..models.native import ArcLM

        return ArcLM(
            vocab_size=int(config.vocab_size),
            embed_dim=int(config.embed_dim),
            block_size=int(config.block_size),
            num_blocks=int(config.num_blocks),
            dropout=float(config.dropout),
        ).to(runtime.torch_device())


__all__ = ["ArcLMNativeCausalLM"]
