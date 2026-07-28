"""Supported-model capability metadata for ArcLM.

The entries in this module describe what ArcLM itself verifies or documents.
They are intentionally narrower than the set of models that external libraries
such as Transformers may be able to load.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Iterable, List, Optional


OFFICIAL = "official"
EXPERIMENTAL = "experimental"
COMPATIBLE_UNTESTED = "compatible_unverified"
NOT_SUPPORTED = "unsupported"


@dataclass(frozen=True)
class ModelCapability:
    """Capability record for a model family or architecture.

    Parameters:
        family: Human-readable model family name.
        example_models: Representative public model IDs or local sources.
        architecture: Architecture category ArcLM expects.
        status: One of the support status constants in this module.
        training: Whether ArcLM documents or verifies training/fine-tuning.
        inference: Whether ArcLM documents or verifies inference.
        quantization: Quantization behavior documented by ArcLM.
        device_requirements: Known CPU/GPU expectations.
        tokenizer_requirements: Requirements for tokenizer loading or metadata.
        attention_implementation: Attention implementation notes.
        precision_support: Precision values accepted by the loading path.
        known_limitations: Important limitations users should know before use.
        verification: Evidence used to assign the support level.
    """

    family: str
    example_models: List[str]
    architecture: str
    status: str
    training: str
    inference: str
    quantization: str = "Not verified by ArcLM."
    device_requirements: str = "Depends on model size and backend."
    tokenizer_requirements: str = "Tokenizer must be loadable or stored in an ArcLM checkpoint."
    attention_implementation: str = "Backend default."
    precision_support: str = "Backend default."
    known_limitations: List[str] = field(default_factory=list)
    verification: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Return a JSON-serializable representation."""

        return asdict(self)


SUPPORTED_MODELS: tuple[ModelCapability, ...] = (
    ModelCapability(
        family="ArcLM native checkpoint",
        example_models=["ArcLM", "MiniGPT"],
        architecture="Compact GPT-style decoder-only causal LM",
        status=OFFICIAL,
        training="Yes",
        inference="Yes",
        quantization="No native quantization support.",
        device_requirements="CPU and CUDA are supported through PyTorch when available.",
        tokenizer_requirements="Word or SentencePiece tokenizer metadata must match the checkpoint.",
        attention_implementation="ArcLM single-head causal self-attention with a triangular mask.",
        precision_support="PyTorch module precision controlled by user code; no high-level mixed-precision API.",
        known_limitations=[
            "Small educational architecture, not a replacement for production-scale transformer stacks.",
            "No multi-head attention setting despite older CLI/config references to num_heads.",
        ],
        verification=[
            "tests/test_library_smoke.py",
            "tests/test_external_engine.py",
            "tests/test_instruction_sft.py",
            "examples/01_quickstart.py",
            "examples/11_inference.py",
        ],
    ),
    ModelCapability(
        family="GPT-2 through Hugging Face",
        example_models=["gpt2", "hf-internal-testing/tiny-random-gpt2"],
        architecture="Decoder-only causal LM loaded with AutoModelForCausalLM",
        status=OFFICIAL,
        training="Yes, for SFT through train_sft when dependencies and hardware are available.",
        inference="Yes, through load_any_model/load_external_for_inference when loading succeeds.",
        quantization="load_in_8bit/load_in_4bit options are passed through; bitsandbytes is not tested here.",
        device_requirements="CPU works for tiny examples; real GPT-2 variants require enough local memory.",
        tokenizer_requirements="AutoTokenizer.from_pretrained must succeed.",
        attention_implementation="Transformers implementation.",
        precision_support="auto, fp32, fp16, bf16 where supported by Transformers and hardware.",
        known_limitations=[
            "Official certification uses hf-internal-testing/tiny-random-gpt2 for automated tests.",
            "The public gpt2 model is documented as the real-world family member but is not downloaded in every test run.",
        ],
        verification=[
            "tests/test_smart_loader.py",
            "tests/test_gpt2_certification.py",
            "examples/08_huggingface_sft.py",
            "examples/09_lora_sft.py",
            "examples/15_custom_hf_sft_loop.py",
        ],
    ),
    ModelCapability(
        family="Qwen through Hugging Face",
        example_models=["Qwen/Qwen3-0.6B"],
        architecture="Decoder-only causal LM loaded with AutoModelForCausalLM",
        status=EXPERIMENTAL,
        training="Yes, documented for SFT example; not automated in the core test suite.",
        inference="Yes, documented in Qwen example scripts; depends on model download and hardware.",
        quantization="No Qwen-specific quantization verification.",
        device_requirements="Qwen3-0.6B requires sufficient CPU/GPU memory and transformers>=4.51.",
        tokenizer_requirements="Qwen tokenizer and chat template must load from Transformers.",
        attention_implementation="Transformers implementation.",
        precision_support="auto, fp32, fp16, bf16 where supported by Transformers and hardware.",
        known_limitations=[
            "Qwen3 example is reproducible but not part of automated CI.",
            "Remote code and chat-template behavior can change with upstream model packages.",
        ],
        verification=[
            "examples/qwen3_0_6b_sft/README.md",
            "examples/qwen3_0_6b_sft/train_qwen3_0_6b_sft.py",
            "examples/qwen3_0_6b_sft/test_base_model.py",
        ],
    ),
    ModelCapability(
        family="Llama through Hugging Face",
        example_models=["hf-internal-testing/tiny-random-LlamaForCausalLM"],
        architecture="Decoder-only causal LM loaded with AutoModelForCausalLM",
        status=EXPERIMENTAL,
        training="Minimal CPU SFT step certified for the tiny random Llama test artifact.",
        inference="CPU inference and deterministic greedy generation certified for the tiny random Llama test artifact.",
        quantization="No Llama-specific quantization verification.",
        device_requirements="CPU works for the tiny certification artifact; real Llama-family models require substantially more memory.",
        tokenizer_requirements="AutoTokenizer.from_pretrained must succeed; chat-template behavior must be validated for conversation data.",
        attention_implementation="Transformers implementation.",
        precision_support="auto and fp32 on CPU; GPU precision is not certified in this environment.",
        known_limitations=[
            "Experimental support is based on hf-internal-testing/tiny-random-LlamaForCausalLM, not production Llama checkpoints.",
            "GPU support, large checkpoints, and chat-template correctness remain unverified.",
        ],
        verification=[
            "arclm.certification.certify_model_family",
            "manual Phase 4 CPU certification run on hf-internal-testing/tiny-random-LlamaForCausalLM",
        ],
    ),
    ModelCapability(
        family="Mistral, Gemma, Falcon through Hugging Face",
        example_models=["owner/model-id"],
        architecture="Expected decoder-only causal LM loaded with AutoModelForCausalLM",
        status=COMPATIBLE_UNTESTED,
        training="Not verified by ArcLM.",
        inference="Compatible path exists, but no family-specific test or example verifies behavior.",
        quantization="Options are passed through only.",
        device_requirements="Depends on the selected model and backend.",
        tokenizer_requirements="AutoTokenizer.from_pretrained must succeed.",
        attention_implementation="Transformers implementation.",
        precision_support="auto, fp32, fp16, bf16 where supported by Transformers and hardware.",
        known_limitations=[
            "The loader may accept these models, but ArcLM does not claim official support yet.",
            "Users must validate tokenizer templates, memory use, and generation quality.",
        ],
        verification=[
            "arclm/external_inference.py model-type detection only",
            "arclm/loaders/smart_loader.py model-type detection only",
        ],
    ),
    ModelCapability(
        family="Encoder-only and sequence-to-sequence models",
        example_models=["bert-base-uncased", "t5-small"],
        architecture="Masked LM or encoder-decoder",
        status=NOT_SUPPORTED,
        training="No",
        inference="No",
        quantization="Out of scope.",
        tokenizer_requirements="Out of scope.",
        attention_implementation="Out of scope.",
        precision_support="Out of scope.",
        known_limitations=[
            "ArcLM's public workflows target causal language modeling.",
            "Use a framework designed for masked-LM or seq2seq tasks instead.",
        ],
        verification=["Project scope decision documented for 0.9.0.dev0."],
    ),
)


def get_supported_models(status: Optional[str] = None) -> List[ModelCapability]:
    """Return ArcLM's declared model-support records.

    Parameters:
        status: Optional support status filter. Use one of ``OFFICIAL``,
            ``EXPERIMENTAL``, ``COMPATIBLE_UNTESTED``, or ``NOT_SUPPORTED``.

    Returns:
        A list of :class:`ModelCapability` entries.

    Raises:
        ValueError: If ``status`` is not a known support status.
    """

    statuses = {OFFICIAL, EXPERIMENTAL, COMPATIBLE_UNTESTED, NOT_SUPPORTED}
    if status is not None and status not in statuses:
        raise ValueError(
            "status must be one of: " + ", ".join(sorted(statuses))
        )
    records = list(SUPPORTED_MODELS)
    if status is not None:
        records = [record for record in records if record.status == status]
    return records


def get_model_capability(family: str) -> ModelCapability:
    """Look up model capability metadata by family name.

    Matching is case-insensitive and accepts substring matches for convenience.
    """

    normalized = family.lower().strip()
    for record in SUPPORTED_MODELS:
        if normalized == record.family.lower() or normalized in record.family.lower():
            return record
    raise KeyError(f"Unknown ArcLM model family: {family}")


def is_model_officially_supported(family: str) -> bool:
    """Return whether a model family is officially supported by ArcLM."""

    return get_model_capability(family).status == OFFICIAL


__all__ = [
    "COMPATIBLE_UNTESTED",
    "EXPERIMENTAL",
    "ModelCapability",
    "NOT_SUPPORTED",
    "OFFICIAL",
    "SUPPORTED_MODELS",
    "get_model_capability",
    "get_supported_models",
    "is_model_officially_supported",
]
