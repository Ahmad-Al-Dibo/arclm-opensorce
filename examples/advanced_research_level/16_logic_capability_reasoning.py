"""Research logic: reason about model capability claims and support status."""

from pathlib import Path
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from arclm.logics import And, Biconditional, Implication, Not, Or, Symbol, model_check


def main():
    has_round_trip_test = Symbol("HasRoundTripTest")
    has_corruption_test = Symbol("HasCorruptionTest")
    has_adapter_reload_test = Symbol("HasAdapterReloadTest")
    native_artifact_supported = Symbol("NativeArtifactSupported")
    lora_supported = Symbol("LoRASupported")
    production_ready = Symbol("ProductionReady")
    experimental = Symbol("Experimental")

    evidence_rules = And(
        Implication(And(has_round_trip_test, has_corruption_test), native_artifact_supported),
        Implication(has_adapter_reload_test, lora_supported),
        Implication(And(native_artifact_supported, lora_supported), production_ready),
        Biconditional(experimental, Not(production_ready)),
    )

    evidence = And(
        evidence_rules,
        has_round_trip_test,
        has_corruption_test,
        has_adapter_reload_test,
    )

    incomplete_evidence = And(
        evidence_rules,
        has_round_trip_test,
        Not(has_corruption_test),
        has_adapter_reload_test,
    )

    print("Rules:", evidence_rules.formula())
    print("Full evidence supports native artifacts?", model_check(evidence, native_artifact_supported))
    print("Full evidence supports LoRA?", model_check(evidence, lora_supported))
    print("Full evidence supports production readiness?", model_check(evidence, production_ready))
    print("Incomplete evidence supports production readiness?", model_check(incomplete_evidence, production_ready))
    print("Incomplete evidence is experimental?", model_check(incomplete_evidence, experimental))
    print("Any artifact-or-adapter support?", Or(native_artifact_supported, lora_supported).formula())


if __name__ == "__main__":
    main()
