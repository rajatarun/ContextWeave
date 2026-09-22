"""Every Ref, GetAtt and Sub in the template names something that exists.

`!Ref KMSKey` shipped in the health resources. The key is called
`ArtifactsKMSKey`, and nothing in this repository could tell: the YAML is valid,
the tests passed, and the only thing that knew was `sam validate --lint` in CI —
which runs after the push, on a runner, minutes later. This file moves that
check offline, where it costs nothing.

It also checks the other half, which cfn-lint reports only as a warning: a
`GetAtt` to a resource that carries a `Condition` is a template error whenever
that condition is false, whether or not the value is used. So a conditional
resource may only be referenced from somewhere equally conditional.
"""
from __future__ import annotations

import pathlib
import re

import pytest
import yaml

REPO = pathlib.Path(__file__).resolve().parents[1]


class CfnLoader(yaml.SafeLoader):
    pass


def _keep(loader, suffix, node):
    if isinstance(node, yaml.ScalarNode):
        value = loader.construct_scalar(node)
    elif isinstance(node, yaml.SequenceNode):
        value = loader.construct_sequence(node, deep=True)
    else:
        value = loader.construct_mapping(node, deep=True)
    return {"fn": suffix, "arg": value}


CfnLoader.add_multi_constructor("!", _keep)
TEMPLATE = yaml.load((REPO / "template.yaml").read_text(), Loader=CfnLoader)

RESOURCES = TEMPLATE["Resources"]
PARAMETERS = TEMPLATE.get("Parameters", {})
CONDITIONS = TEMPLATE.get("Conditions", {})
MAPPINGS = TEMPLATE.get("Mappings", {})

# Pseudo-parameters are always resolvable and are declared nowhere.
PSEUDO = {"AWS::AccountId", "AWS::NoValue", "AWS::NotificationARNs", "AWS::Partition",
          "AWS::Region", "AWS::StackId", "AWS::StackName", "AWS::URLSuffix"}
KNOWN = set(RESOURCES) | set(PARAMETERS) | PSEUDO

SUB_VAR = re.compile(r"\$\{([^}!][^}]*)\}")


def _walk(node, path="", guards=frozenset()):
    """Yield (logical_name, kind, path, guards) for every reference.

    `guards` is the set of conditions that must be true for this position to be
    evaluated at all -- the true-branch of every enclosing `Fn::If`. That is the
    whole point: CloudFormation never evaluates the branch it does not take, so
    a reference to a conditional resource is safe exactly when it sits under a
    guard for that condition. Matching on the path string instead gets this
    wrong, because the loader turns `!If` into an ordinary `arg` key.
    """
    if isinstance(node, dict):
        fn = node.get("fn")
        arg = node.get("arg")
        if fn == "Ref" and isinstance(arg, str):
            yield arg, "Ref", path, guards
            return
        if fn == "GetAtt":
            name = arg.split(".")[0] if isinstance(arg, str) else (
                arg[0] if isinstance(arg, list) and arg else None)
            if isinstance(name, str):
                yield name, "GetAtt", path, guards
            return
        if fn == "If" and isinstance(arg, list) and len(arg) == 3:
            condition = arg[0] if isinstance(arg[0], str) else None
            taken = guards | {condition} if condition else guards
            yield from _walk(arg[1], f"{path}/If[{condition}]", taken)
            # The false branch is not guarded by the condition, and whatever
            # guarantees it needs are its own.
            yield from _walk(arg[2], f"{path}/If[not {condition}]", guards)
            return
        if fn == "Sub":
            body = arg[0] if isinstance(arg, list) else arg
            declared = set()
            if isinstance(arg, list) and len(arg) > 1 and isinstance(arg[1], dict):
                declared = set(arg[1])
                # The substitution map's own values hold real references.
                yield from _walk(arg[1], f"{path}/Sub-vars", guards)
            if isinstance(body, str):
                for var in SUB_VAR.findall(body):
                    name = var.split(".")[0].strip()
                    if name and name not in declared:
                        yield name, "Sub", path, guards
            return
        for key, value in node.items():
            yield from _walk(value, f"{path}/{key}", guards)
    elif isinstance(node, list):
        for i, value in enumerate(node):
            yield from _walk(value, f"{path}[{i}]", guards)


ALL_REFS = list(_walk(TEMPLATE))


def test_the_template_has_references_to_check():
    """A walker that silently matches nothing passes every assertion below."""
    assert len(ALL_REFS) > 100, f"only found {len(ALL_REFS)} references"
    assert any(kind == "GetAtt" for _, kind, _, _ in ALL_REFS)
    assert any(kind == "Sub" for _, kind, _, _ in ALL_REFS)
    assert any(guards for *_, guards in ALL_REFS), \
        "no reference sits under an Fn::If, so the guard tracking is untested"


def test_every_reference_names_something_declared():
    unknown = sorted({(name, path) for name, _, path, _ in ALL_REFS if name not in KNOWN})
    assert not unknown, (
        "reference to a resource or parameter that does not exist:\n  "
        + "\n  ".join(f"{name}  at {path}" for name, path in unknown))


# ── conditional resources may only be referenced conditionally ──────────────

def _owning_resource(path: str) -> str | None:
    parts = [p for p in path.split("/") if p]
    if len(parts) >= 2 and parts[0] == "Resources":
        return parts[1]
    return None


CONDITIONAL = {name: res["Condition"] for name, res in RESOURCES.items()
               if "Condition" in res}


def _implies(have: str, need: str) -> bool:
    """Does condition `have` being true guarantee `need` is true?

    Only the shape this template uses is decided: equality, and an `Fn::And`
    with the needed condition among its conjuncts (`CreateHealthBucket` is
    `HealthEnabled` and an empty bucket name, so it implies it). Anything else
    is reported rather than assumed -- an unproven implication is a reference
    somebody should look at, not one to wave through.
    """
    if have == need:
        return True
    definition = CONDITIONS.get(have)
    if not isinstance(definition, dict) or definition.get("fn") != "And":
        return False
    for conjunct in definition.get("arg") or []:
        if isinstance(conjunct, dict) and conjunct.get("fn") == "Condition":
            if _implies(conjunct["arg"], need):
                return True
    return False


def test_conditional_resources_are_not_referenced_unconditionally():
    """CloudFormation resolves a GetAtt before it knows whether the value is
    used, so referencing a condition-false resource fails the whole stack --
    which is how the health surface made every deploy undeployable while the
    lint reported it as a warning.
    """
    assert CONDITIONAL, "no conditional resources: the check would be vacuous"
    offenders = []
    for name, kind, path, guards in ALL_REFS:
        if name not in CONDITIONAL or kind == "Sub":
            continue
        needed = CONDITIONAL[name]
        if any(_implies(guard, needed) for guard in guards):
            continue
        owner = _owning_resource(path)
        if owner is None:                      # Outputs carry their own Condition
            continue
        owner_condition = RESOURCES.get(owner, {}).get("Condition")
        if not (owner_condition and _implies(owner_condition, needed)):
            offenders.append(f"{owner} -> {name} (needs {needed}) at {path}")
    assert not offenders, "unconditional reference to a conditional resource:\n  " + \
        "\n  ".join(offenders)


def test_the_authorizer_invoke_role_is_deliberately_unconditional():
    """The one exception, and it has to be: the API's authorizer block GetAtts
    this role, and SAM rejects `Fn::If` anywhere inside `Auth` -- on the block,
    on `Authorizers`, or on the authorizer. `sam validate --lint` accepts the
    intrinsic and the transform then fails, so CI goes green and `sam build`
    dies."""
    assert "Condition" not in RESOURCES["HealthAuthorizerInvokeRole"], (
        "conditioning this role makes the stack undeployable whenever the "
        "health surface is off, which is every deploy without an authorizer")


def test_the_authorizer_resolves_when_the_health_surface_is_off():
    """Both places that name the authorizer function: an empty string is not a
    valid IAM policy Resource and not a valid authorizer URI."""
    api = TEMPLATE["Resources"]["ExpertiseAPI"]["Properties"]
    arn = api["Auth"]["Authorizers"]["SiweAuthorizer"]["FunctionArn"]
    assert arn.get("fn") == "If", "FunctionArn is empty when HealthEnabled is false"
    assert arn["arg"][0] == "HealthEnabled"

    role = RESOURCES["HealthAuthorizerInvokeRole"]["Properties"]["Policies"][0]
    resource = role["PolicyDocument"]["Statement"][0]["Resource"]
    assert resource.get("fn") == "If", "empty policy Resource when HealthEnabled is false"
    assert resource["arg"][0] == "HealthEnabled"


# ── the parameter has to actually be passed ─────────────────────────────────

WORKFLOW = (REPO / ".github" / "workflows" / "deploy.yaml").read_text()


def test_the_deploy_passes_the_authorizer_parameter():
    """Gated on a parameter nothing supplied, every health resource was
    condition-false on every deploy: the bucket, the database, the endpoint.
    Infrastructure for nothing, deploying green."""
    assert "SiweAuthorizerFunctionArn=" in WORKFLOW, \
        "the health surface is gated on a parameter the deploy never passes"


def test_the_deploy_reads_the_authorizer_from_a_stack_output():
    """What this proves: the ARN is read from a CloudFormation output rather
    than hardcoded. What it cannot prove: that `siwe-infra` is the right stack.
    A wrong name and an undeployed sibling both resolve to nothing from here --
    which is why the step warns by name instead of failing, and why the name
    is taken from TeamWeave's deploy, the other consumer of the same output.
    """
    assert re.search(r"--stack-name\s+siwe-infra\b", WORKFLOW), \
        "the authorizer stack is not named in a describe-stacks call"
    assert re.search(r"OutputKey=='AuthorizerFunctionArn'", WORKFLOW)


def test_a_missing_authorizer_is_not_a_hardcoded_none():
    """`aws cloudformation describe-stacks --output text` prints the string
    None for an absent output, and None is not empty -- it would pass the
    template's non-empty test and enable the health surface against an
    authorizer ARN that is the word None."""
    assert '"$SIWE_ARN" = "None"' in WORKFLOW, \
        "the literal None from the CLI would enable the health surface"


def test_the_implication_rule_is_not_a_yes_machine():
    """A `_implies` that returned True would silence the check above entirely."""
    assert _implies("CreateHealthBucket", "HealthEnabled"), \
        "CreateHealthBucket is an And over HealthEnabled"
    assert not _implies("HealthEnabled", "CreateHealthBucket"), \
        "the implication does not run backwards"
    assert not _implies("HealthEnabled", "NoSuchCondition")
