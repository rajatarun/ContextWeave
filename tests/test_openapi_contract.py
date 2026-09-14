"""Keep openapi/contextweave.yaml honest against the contract and the stack.

There are now three descriptions of the same HTTP surface, and two of them are
only useful if they cannot drift:

* ``contracts/contextweave_http_api.json`` -- the source of truth. TeamWeave
  validates its client against this exact file, and ContextWeave validates its
  real handlers against it in ``tests/test_http_api_contract.py``.
* ``openapi/contextweave.yaml`` -- the machine-readable spec an automation
  harness or a code generator consumes.
* ``scripts/stack_env.py`` -- the names of the stack Outputs a harness reads to
  find the API and the database, rather than hardcoding either.

A spec that says a field is optional when the contract requires it is worse
than no spec: a generated client drops the field and nothing complains until
production. So this test never retypes a key list. It reads the contract's
``required_*_keys`` and asserts the spec's ``required:`` arrays are the same
sets, and it validates the contract's own samples against the spec's schemas --
which catches the other direction, a spec that requires something the handlers
do not actually return.

The third test is the one that makes ``stack_env.py`` trustworthy: every stack
Output it reads must exist in ``template.yaml``. Deleting or renaming an output
then fails here, at the commit that does it, instead of at the next automation
run against a deployed stack.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
SPEC_PATH = REPO_ROOT / "openapi" / "contextweave.yaml"
CONTRACT_PATH = REPO_ROOT / "contracts" / "contextweave_http_api.json"
TEMPLATE_PATH = REPO_ROOT / "template.yaml"

sys.path.insert(0, str(REPO_ROOT / "scripts"))
import stack_env  # noqa: E402


@pytest.fixture(scope="module")
def spec() -> dict:
    with open(SPEC_PATH, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


@pytest.fixture(scope="module")
def contract() -> dict:
    with open(CONTRACT_PATH, encoding="utf-8") as fh:
        return json.load(fh)


def _ok_schema(spec: dict, path: str, method: str) -> dict:
    """The 200 response schema for one operation."""
    op = spec["paths"][path][method]
    return op["responses"]["200"]["content"]["application/json"]["schema"]


def _branch_with(schema: dict, key: str) -> dict:
    """Pick the oneOf branch whose required list contains ``key``.

    /routing-decisions returns two different shapes off one operation
    (mode=list vs mode=summary), so the spec models it as a oneOf and the
    contract lists them as two separate endpoints.
    """
    if "oneOf" not in schema:
        return schema
    for branch in schema["oneOf"]:
        if key in branch.get("required", []):
            return branch
    raise AssertionError(f"no oneOf branch requires {key!r}")


def _to_json_schema(node):
    """Strip the OpenAPI 3.0 dialect down to something jsonschema can run.

    Only ``nullable: true`` differs in what we use: OpenAPI spells an optional
    null as a sibling boolean, JSON Schema spells it as a type union. Without
    this, every legitimately-null field in the contract samples (avgRating,
    ratedAt) would fail validation and the test would be asserting a falsehood.
    """
    if isinstance(node, list):
        return [_to_json_schema(v) for v in node]
    if not isinstance(node, dict):
        return node
    out = {k: _to_json_schema(v) for k, v in node.items() if k != "nullable"}
    if node.get("nullable") is True and "type" in out:
        t = out["type"]
        out["type"] = [t, "null"] if isinstance(t, str) else list(t) + ["null"]
    return out


# contract endpoint -> (spec path, method, key identifying the oneOf branch)
ENDPOINT_MAP = {
    "GET /health": ("/health", "get", None),
    "POST /query-expertise": ("/query-expertise", "post", None),
    "POST /feedback": ("/feedback", "post", None),
    "GET /routing-decisions?mode=summary": ("/routing-decisions", "get", "groups"),
    "GET /routing-decisions?mode=list": ("/routing-decisions", "get", "items"),
}


def test_every_contract_endpoint_is_in_the_spec(contract, spec):
    """A contracted endpoint the spec omits is invisible to any generated client."""
    assert set(ENDPOINT_MAP) == set(contract["endpoints"]), (
        "ENDPOINT_MAP and the contract's endpoints disagree -- an endpoint was "
        "added or renamed in one place only"
    )
    for name, (path, method, _) in ENDPOINT_MAP.items():
        assert path in spec["paths"], f"{name}: openapi spec has no path {path}"
        assert method in spec["paths"][path], f"{name}: no {method.upper()} on {path}"


@pytest.mark.parametrize("endpoint", sorted(ENDPOINT_MAP))
def test_spec_required_keys_match_the_contract(endpoint, contract, spec):
    """The spec's `required:` arrays are the contract's key lists, exactly."""
    path, method, branch_key = ENDPOINT_MAP[endpoint]
    entry = contract["endpoints"][endpoint]
    schema = _branch_with(_ok_schema(spec, path, method), branch_key) if branch_key \
        else _ok_schema(spec, path, method)

    assert set(schema.get("required", [])) == set(entry["required_top_level_keys"]), (
        f"{endpoint}: spec requires {sorted(schema.get('required', []))}, "
        f"contract requires {sorted(entry['required_top_level_keys'])}"
    )

    # Nested lists, where the contract pins the shape of the repeated element.
    nested = [
        ("required_source_keys", ("sources",)),
        ("required_group_keys", ("groups",)),
        ("required_item_keys", ("items",)),
    ]
    for contract_key, prop_path in nested:
        if contract_key not in entry:
            continue
        node = schema
        for prop in prop_path:
            node = node["properties"][prop]
        item_schema = node["items"]
        assert set(item_schema.get("required", [])) == set(entry[contract_key]), (
            f"{endpoint}: {'.'.join(prop_path)}[] requires "
            f"{sorted(item_schema.get('required', []))}, contract requires "
            f"{sorted(entry[contract_key])}"
        )


@pytest.mark.parametrize("endpoint", sorted(ENDPOINT_MAP))
def test_contract_samples_validate_against_the_spec(endpoint, contract, spec):
    """The other direction: the spec must accept what the handlers really return.

    The contract samples are the same payloads tests/test_http_api_contract.py
    checks the live handlers against, so a spec that rejects one is a spec that
    would reject production traffic.
    """
    jsonschema = pytest.importorskip("jsonschema")
    path, method, branch_key = ENDPOINT_MAP[endpoint]
    schema = _branch_with(_ok_schema(spec, path, method), branch_key) if branch_key \
        else _ok_schema(spec, path, method)
    jsonschema.validate(
        instance=contract["endpoints"][endpoint]["sample"],
        schema=_to_json_schema(schema),
    )


def _template_output_names() -> set:
    """Top-level keys of template.yaml's Outputs block.

    Parsed with a regex rather than yaml.safe_load because the template is full
    of !Sub/!Ref tags a safe loader rejects, and a permissive loader that maps
    unknown tags to None silently reports every !Sub-valued field as absent --
    which is exactly how this kind of check ends up asserting nothing.
    """
    text = TEMPLATE_PATH.read_text(encoding="utf-8")
    block = text.split("\nOutputs:\n", 1)
    assert len(block) == 2, "template.yaml has no top-level Outputs: block"
    names = set()
    for line in block[1].splitlines():
        if line and not line.startswith(" ") and not line.startswith("#"):
            break  # next top-level section
        m = re.match(r"^  ([A-Za-z][A-Za-z0-9]*):\s*$", line)
        if m:
            names.add(m.group(1))
    return names


def test_stack_env_reads_only_outputs_the_template_publishes():
    """Automation resolves everything from stack outputs; those must exist."""
    published = _template_output_names()
    assert published, "parsed no outputs from template.yaml -- the parser is broken"
    wanted = set(stack_env.REQUIRED_OUTPUTS.values()) | set(stack_env.OPTIONAL_OUTPUTS.values())
    missing = sorted(wanted - published)
    assert not missing, (
        f"scripts/stack_env.py reads stack outputs that template.yaml does not "
        f"publish: {missing}. Either add the Output or stop reading it -- a "
        f"harness cannot resolve what the stack does not export."
    )


def test_required_outputs_cover_api_and_database():
    """The point of the exercise: no harness should need a hardcoded coordinate.

    API base URL, and enough to reach Postgres: host, port, database name, the
    secret with the credentials, and the instance identifier needed to start a
    stopped instance before a run.
    """
    required = set(stack_env.REQUIRED_OUTPUTS.values())
    for needed in ("APIEndpoint", "PostgresEndpoint", "PostgresPort",
                   "PostgresDbName", "PostgresSecretArn", "PostgresInstanceIdentifier"):
        assert needed in required, f"{needed} is not resolved from the stack"


def test_openapi_spec_path_output_points_at_the_spec():
    """template.yaml advertises where the spec lives; it must actually be there."""
    text = TEMPLATE_PATH.read_text(encoding="utf-8")
    m = re.search(r"^  OpenApiSpecPath:\n(?:.*\n)*?^    Value:\s*(\S+)\s*$", text, re.M)
    assert m, "template.yaml has no OpenApiSpecPath output with a literal Value"
    assert (REPO_ROOT / m.group(1)).is_file(), (
        f"OpenApiSpecPath output points at {m.group(1)}, which does not exist"
    )


def test_engine_version_is_greppable_by_the_deploy_workflow():
    """The pre-deploy guard reads EngineVersion out of the template with awk.

    That guard exists because the pin goes stale on its own -- RDS auto-upgrades
    the minor version in a maintenance window -- and a stale pin is not a no-op:
    minor versions cannot be downgraded, so CloudFormation spends three minutes
    building a changeset, failing on PostgresRDS and rolling the stack back to
    tell you one number is wrong.

    The guard matches `^      EngineVersion: '...'`. If the property is
    reindented or rewritten the awk returns nothing, and while the step does
    fail loudly on that, it fails during a deploy. This fails at the commit.
    """
    lines = re.findall(r"^      EngineVersion: '([^']+)'$",
                       TEMPLATE_PATH.read_text(encoding="utf-8"), re.M)
    assert len(lines) == 1, (
        f"expected exactly one 6-space-indented EngineVersion line for the "
        f"deploy workflow's awk to find, got {lines}. If the property moved, "
        f"update the 'Check the template's EngineVersion against the live "
        f"instance' step in .github/workflows/deploy.yaml too."
    )
