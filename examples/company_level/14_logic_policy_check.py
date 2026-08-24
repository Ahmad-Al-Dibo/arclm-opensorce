"""Company logic: check deployment policy gates with symbolic rules."""

from pathlib import Path
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from arclm.logics import And, Implication, Not, Or, Symbol, model_check


def main():
    tests_pass = Symbol("TestsPass")
    artifact_signed = Symbol("ArtifactSigned")
    data_approved = Symbol("DataApproved")
    can_deploy = Symbol("CanDeploy")
    needs_review = Symbol("NeedsReview")

    policy = And(
        Implication(And(tests_pass, artifact_signed, data_approved), can_deploy),
        Implication(Or(Not(tests_pass), Not(data_approved)), needs_review),
        tests_pass,
        artifact_signed,
        data_approved,
    )

    print("Policy:", policy.formula())
    print("Can deploy?", model_check(policy, can_deploy))
    print("Needs review?", model_check(policy, needs_review))

    blocked_policy = And(
        Implication(And(tests_pass, artifact_signed, data_approved), can_deploy),
        Implication(Or(Not(tests_pass), Not(data_approved)), needs_review),
        tests_pass,
        artifact_signed,
        Not(data_approved),
    )

    print("Blocked policy can deploy?", model_check(blocked_policy, can_deploy))
    print("Blocked policy needs review?", model_check(blocked_policy, needs_review))


if __name__ == "__main__":
    main()
