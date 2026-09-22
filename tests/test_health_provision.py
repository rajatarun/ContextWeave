"""The provisioning that makes the isolation a grant rather than a sentence.

`health_db` refuses to fall back to the expertise credential, and the template
gives the health surface its own secret -- but neither creates a database. Until
this runs, `healthrag` does not exist and every one of those guarantees is about
a thing that is not there.

The property under test is narrow and worth stating exactly: after this runs,
the role the expertise retrieval path connects as cannot `CONNECT` to the health
database. Everything else here exists to make that survive a second deploy, a
rotated password, and a connection handed over mid-transaction.
"""
from __future__ import annotations

import pathlib
import sys

import pytest
import yaml

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.shared import health_provision  # noqa: E402


# ── a fake that fails the way psycopg2 fails ────────────────────────────────

class FakeCursor:
    def __init__(self, conn):
        self.conn = conn

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def execute(self, sql, params=None):
        if not self.conn.autocommit:
            # psycopg2 opens a transaction on the first statement and holds it
            # until commit or rollback. That is the state that makes the
            # autocommit assignment raise, so the fake has to reproduce it.
            self.conn.in_transaction = True
        self.conn.statements.append((" ".join(sql.split()), params))
        self._result = self.conn.answer(sql, params)

    def fetchone(self):
        return self._result


class FakeConn:
    """Models the two psycopg2 behaviours this code has to respect: a statement
    opens a transaction, and `autocommit` cannot be assigned while one is open.
    """

    def __init__(self, *, roles=(), databases=(), can_connect=False):
        self._autocommit = False
        self.in_transaction = False
        self.statements: list[tuple[str, tuple | None]] = []
        self.roles = set(roles)
        self.databases = set(databases)
        self.can_connect = can_connect
        self.rollbacks = 0

    @property
    def autocommit(self):
        return self._autocommit

    @autocommit.setter
    def autocommit(self, value):
        if self.in_transaction:
            raise RuntimeError(
                "set_session cannot be used inside a transaction"
            )
        self._autocommit = value

    def rollback(self):
        self.rollbacks += 1
        self.in_transaction = False

    def cursor(self):
        return FakeCursor(self)

    def answer(self, sql, params):
        if "pg_roles" in sql:
            return (1,) if params[0] in self.roles else None
        if "pg_database" in sql:
            return (1,) if params[0] in self.databases else None
        if "has_database_privilege" in sql:
            return (self.can_connect,)
        if "current_user" in sql:
            return ("expertiserag",)
        return None


SECRET = {"username": "healthrag", "dbname": "healthrag", "password": "s3cret"}


@pytest.fixture(autouse=True)
def _secret(monkeypatch):
    monkeypatch.setattr(health_provision, "_secret", lambda arn: dict(SECRET))
    monkeypatch.setenv("HEALTH_POSTGRES_SECRET_ARN", "arn:aws:secret:health")


def sql_of(conn):
    return [s for s, _ in conn.statements]


# ── the property the store exists for ───────────────────────────────────────

def test_public_is_revoked_before_the_health_role_is_granted():
    """Order is the whole thing. A new database grants CONNECT to PUBLIC and
    every role is a member of PUBLIC, so a REVOKE aimed only at the expertise
    role leaves it connecting through a grant nobody wrote down. Granting the
    health role first and revoking PUBLIC afterwards would also revoke the
    grant just made."""
    conn = FakeConn()
    health_provision.provision(conn)
    sql = sql_of(conn)

    revoke = next(i for i, s in enumerate(sql) if s.startswith("REVOKE ALL ON DATABASE"))
    grant = next(i for i, s in enumerate(sql) if s.startswith("GRANT CONNECT ON DATABASE"))
    create = next(i for i, s in enumerate(sql) if s.startswith("CREATE DATABASE"))

    assert "FROM PUBLIC" in sql[revoke], "the revoke must name PUBLIC, not a role"
    assert create < revoke < grant


def test_revoke_names_public_and_not_merely_the_expertise_role():
    conn = FakeConn()
    health_provision.provision(conn)
    revokes = [s for s in sql_of(conn) if s.startswith("REVOKE")]
    assert revokes, "nothing revoked at all"
    assert all("PUBLIC" in s for s in revokes)


def test_isolation_is_verified_by_asking_postgres():
    """`has_database_privilege` accounts for membership in PUBLIC. A check that
    read `pg_database`'s ACL column by hand would report the expertise role as
    denied while PUBLIC let it straight in."""
    conn = FakeConn(databases={"healthrag"}, can_connect=False)
    result = health_provision.verify_isolation(conn, "expertiserag")
    assert result["isolated"] is True
    assert any("has_database_privilege" in s for s in sql_of(conn))


def test_isolation_fails_loudly_when_the_expertise_role_can_still_connect():
    conn = FakeConn(databases={"healthrag"}, can_connect=True)
    result = health_provision.verify_isolation(conn, "expertiserag")
    assert result["isolated"] is False
    assert "expertiserag" in result["reason"]


def test_isolation_is_not_claimed_when_the_database_does_not_exist():
    """"Nothing can connect to it" is true of a database that was never
    created, and reporting that as isolation would pass the check on exactly
    the deployment where provisioning failed."""
    conn = FakeConn(databases=set())
    result = health_provision.verify_isolation(conn, "expertiserag")
    assert result["isolated"] is False


# ── surviving the second deploy ─────────────────────────────────────────────

def test_a_connection_mid_transaction_is_rolled_back_first():
    """`db_clients.get_pg_connection` pings with a bare `SELECT 1` and never
    commits, so the connection this is handed always arrives inside a
    transaction -- and psycopg2 refuses to change autocommit there. Without the
    rollback this raises before a single statement runs."""
    conn = FakeConn()
    conn.in_transaction = True
    health_provision.provision(conn)   # must not raise
    assert conn.rollbacks >= 1


def test_the_callers_autocommit_setting_is_restored():
    conn = FakeConn()
    assert conn.autocommit is False
    health_provision.provision(conn)
    assert conn.autocommit is False, "left in autocommit, so the next caller's commits are no-ops"


def test_create_database_runs_outside_a_transaction():
    """Postgres rejects CREATE DATABASE inside a transaction block, so it is
    only reachable with autocommit on."""
    conn = FakeConn()
    seen = {}
    real = FakeCursor.execute

    def spy(self, sql, params=None):
        if sql.startswith("CREATE DATABASE"):
            seen["autocommit"] = self.conn.autocommit
        return real(self, sql, params)

    FakeCursor.execute = spy
    try:
        health_provision.provision(conn)
    finally:
        FakeCursor.execute = real
    assert seen.get("autocommit") is True


def test_second_run_creates_nothing_and_still_reasserts_the_grants():
    """Idempotent, and the grants are re-applied rather than skipped: someone
    granting PUBLIC back by hand is repaired by the next deploy."""
    conn = FakeConn(roles={"healthrag"}, databases={"healthrag"})
    result = health_provision.provision(conn)
    sql = sql_of(conn)
    assert not any(s.startswith("CREATE DATABASE") for s in sql)
    assert not any(s.startswith("CREATE ROLE") for s in sql)
    assert any(s.startswith("REVOKE ALL ON DATABASE") for s in sql)
    assert any(s.startswith("GRANT CONNECT ON DATABASE") for s in sql)
    assert result["status"] == "success"


def test_an_existing_role_has_its_password_synced():
    """The secret can be rotated. A role left on the old password makes the
    store unreachable, and the failure reads as a network problem."""
    conn = FakeConn(roles={"healthrag"}, databases={"healthrag"})
    health_provision.provision(conn)
    alters = [(s, p) for s, p in conn.statements if s.startswith("ALTER ROLE")]
    assert alters, "a rotated secret would leave the role on its old password"
    assert alters[0][1] == ("s3cret",)


def test_the_password_is_never_interpolated_into_sql():
    """Passed as a bound parameter, so a generated password containing a quote
    cannot end the statement."""
    conn = FakeConn()
    health_provision.provision(conn)
    assert not any("s3cret" in s for s in sql_of(conn))


# ── refusals ────────────────────────────────────────────────────────────────

def test_no_secret_is_skipped_not_failed(monkeypatch):
    """The health surface is gated on an authorizer. With it off there are no
    health resources to provision, and a deploy must not fail over it."""
    monkeypatch.delenv("HEALTH_POSTGRES_SECRET_ARN", raising=False)
    assert health_provision.provision(FakeConn())["status"] == "skipped"


def test_a_secret_without_a_password_fails_rather_than_creating_a_passwordless_role(monkeypatch):
    monkeypatch.setattr(health_provision, "_secret",
                        lambda arn: {"username": "healthrag", "dbname": "healthrag"})
    conn = FakeConn()
    result = health_provision.provision(conn, secret_arn="arn:x")
    assert result["status"] == "failed"
    assert conn.statements == [], "a role was touched despite having no password to set"


@pytest.mark.parametrize("bad", ["health-rag", 'health"rag', "health rag", "health;drop"])
def test_an_identifier_that_is_not_a_bare_name_is_refused(bad):
    """These are interpolated into the statement text -- a database name cannot
    be a bound parameter -- so the constraint is enforced rather than assumed."""
    with pytest.raises(ValueError):
        health_provision._quote_ident(bad)


# ── the wiring, which is invisible from the module ──────────────────────────

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


def _flatten(node):
    out = []
    if isinstance(node, dict):
        for v in node.values():
            out += _flatten(v)
    elif isinstance(node, list):
        for v in node:
            out += _flatten(v)
    elif isinstance(node, str):
        out.append(node)
    return out


def test_db_init_is_given_the_health_secret():
    env = RESOURCES["DBInitFunction"]["Properties"]["Environment"]["Variables"]
    assert "HEALTH_POSTGRES_SECRET_ARN" in env, \
        "the provisioner reads this and skips silently without it"
    assert "HealthPostgresSecret" in _flatten(env["HEALTH_POSTGRES_SECRET_ARN"])


def test_db_init_may_read_the_health_secret():
    policies = RESOURCES["DBInitRole"]["Properties"]["Policies"]
    assert "HealthPostgresSecret" in _flatten(policies), \
        "the env var names a secret the role cannot read"


def test_the_health_secret_is_not_in_the_shared_secrets_policy():
    """The one-line version of the grant above would hand the health credential
    to the preprocessor and the query API as well -- the two functions the
    separate database exists to keep away from it."""
    shared = RESOURCES["SecretsReadPolicy"]["Properties"]["PolicyDocument"]
    assert "HealthPostgresSecret" not in _flatten(shared)


def test_the_health_secret_generates_a_password():
    """`GenerateSecretKey` is not a property of `GenerateSecretString`; the
    correct name is `GenerateStringKey`, and the wrong one fails the stack."""
    gen = RESOURCES["HealthPostgresSecret"]["Properties"]["GenerateSecretString"]
    assert "GenerateStringKey" in gen
    assert "GenerateSecretKey" not in gen


@pytest.mark.parametrize("fn", ["HealthIngestFunction", "HealthAPIFunction"])
def test_health_functions_reach_the_database(fn):
    """The Postgres instance is not publicly accessible. A function outside the
    VPC resolves its endpoint and then hangs until the connect timeout."""
    props = RESOURCES[fn]["Properties"]
    assert "VpcConfig" in props, f"{fn} cannot route to the database"
    env = props["Environment"]["Variables"]
    assert "HEALTH_POSTGRES_HOST" in env, \
        f"{fn} has no host: the health secret has no target attachment"


def test_the_health_secret_has_no_target_attachment():
    """Deliberate. A `SecretTargetAttachment` writes the *instance's*
    connection details into the secret, and a secret that answers `dbname`
    with the expertise database would point the health store straight at the
    corpus it exists to stay out of."""
    for name, res in RESOURCES.items():
        if res.get("Type") == "AWS::SecretsManager::SecretTargetAttachment":
            assert "HealthPostgresSecret" not in _flatten(res.get("Properties", {})), name


# ── the custom resource, where a failure has to be fatal ────────────────────

def _load_dbinit():
    """Imported by path: `src/` is the Lambda's root, so the module's own
    `from shared.demo_logging import ...` only resolves with `src` on the path.
    """
    import importlib
    sys.path.insert(0, str(REPO / "src"))
    sys.path.insert(0, str(REPO / "src" / "shared"))
    mod = importlib.import_module("ingestion_trigger.handler")
    return importlib.reload(mod)


DBINIT = _load_dbinit()


@pytest.fixture
def dbinit(monkeypatch):
    monkeypatch.setenv("HEALTH_POSTGRES_SECRET_ARN", "arn:aws:secret:health")
    monkeypatch.setattr(DBINIT, "init_db_schema",
                        lambda: {"status": "success", "message": "ok"})
    monkeypatch.setattr(DBINIT, "seed_routing_graph_memgraph",
                        lambda: {"status": "success", "nodes": 1, "edges": 1})
    sent = {}
    monkeypatch.setattr(DBINIT, "_cfn_send",
                        lambda e, c, status, pid, data=None, reason="": sent.update(
                            status=status, reason=reason, data=data or {}))
    return sent


CFN_EVENT = {"RequestType": "Create", "ResponseURL": "", "StackId": "s",
             "RequestId": "r", "LogicalResourceId": "l"}


def test_a_failed_health_provision_fails_the_stack(monkeypatch, dbinit):
    """Not a warning. The health API is live either way, so a database that
    was never created means a person uploads a medical record and the
    ingestion writes it nowhere -- with every signal saying the deploy worked.
    """
    monkeypatch.setattr(DBINIT, "provision_health_db",
                        lambda: {"status": "failed", "reason": "permission denied"})
    result = DBINIT.lambda_handler(dict(CFN_EVENT), None)
    assert result["status"] == "failed"
    assert dbinit["status"] == "FAILED"
    assert "permission denied" in dbinit["reason"]


def test_a_skipped_health_provision_is_not_a_failure(monkeypatch, dbinit):
    """The surface is gated on an authorizer being configured. A stack without
    one has no health resources at all and must still deploy."""
    monkeypatch.setattr(DBINIT, "provision_health_db",
                        lambda: {"status": "skipped", "reason": "not enabled"})
    result = DBINIT.lambda_handler(dict(CFN_EVENT), None)
    assert result["status"] == "success"
    assert dbinit["status"] == "SUCCESS"


def test_the_expertise_role_is_read_from_the_connection_not_configured(monkeypatch):
    """A role name from a parameter is exactly the value that is true in the
    template and false in the database. `current_user` is the role the
    retrieval path actually connects as, which is the only one worth checking.
    """
    conn = FakeConn(databases={"healthrag"}, can_connect=False)
    asked = {}

    class Stub:
        HEALTH_DB = "healthrag"

        @staticmethod
        def provision(c, **kw):
            return {"status": "success", "database": "healthrag"}

        @staticmethod
        def verify_isolation(c, role, dbname=""):
            asked["role"] = role
            return {"isolated": True}

    monkeypatch.setenv("HEALTH_POSTGRES_SECRET_ARN", "arn:x")
    monkeypatch.setattr(DBINIT, "_get_shared_module", lambda name: (
        Stub if name == "health_provision"
        else type("D", (), {"get_pg_connection": staticmethod(lambda: conn)})))
    DBINIT.provision_health_db()
    assert asked["role"] == "expertiserag"
    assert any("current_user" in s for s in sql_of(conn))


def test_a_successful_provision_that_is_not_isolated_is_reported_as_failed(monkeypatch):
    """The statements can all succeed and the property still not hold -- a
    pre-existing database someone granted PUBLIC on is the obvious case."""
    conn = FakeConn(databases={"healthrag"}, can_connect=True)

    class Stub:
        @staticmethod
        def provision(c, **kw):
            return {"status": "success", "database": "healthrag"}

        @staticmethod
        def verify_isolation(c, role, dbname=""):
            return {"isolated": False, "reason": f"{role} can still CONNECT"}

    monkeypatch.setenv("HEALTH_POSTGRES_SECRET_ARN", "arn:x")
    monkeypatch.setattr(DBINIT, "_get_shared_module", lambda name: (
        Stub if name == "health_provision"
        else type("D", (), {"get_pg_connection": staticmethod(lambda: conn)})))
    result = DBINIT.provision_health_db()
    assert result["status"] == "failed"


def test_provisioning_is_skipped_without_the_secret(monkeypatch):
    monkeypatch.delenv("HEALTH_POSTGRES_SECRET_ARN", raising=False)
    assert DBINIT.provision_health_db()["status"] == "skipped"
