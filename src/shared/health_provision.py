"""Create the health database, its role, and the grants that isolate it.

`CREATE DATABASE` is not something a schema migration can do: it cannot run
inside a transaction, and it has to be issued from a connection to a *different*
database. So this runs once from the deploy's custom resource, against the
expertise (admin) connection, and everything it does is idempotent -- Postgres
has no `IF NOT EXISTS` for databases or roles, so each is checked first.

**The revoke that matters is from PUBLIC, not from the expertise role.** A new
database grants `CONNECT` to `PUBLIC`, and every role is a member of PUBLIC. So
revoking from `expertiserag` alone leaves it able to connect anyway, through a
grant nobody wrote down. The order below is deliberate:

    CREATE DATABASE healthrag
    REVOKE ALL ON DATABASE healthrag FROM PUBLIC   <- the one that isolates
    GRANT CONNECT ON DATABASE healthrag TO healthrag

After that, `expertiserag` cannot connect at all, so no query from the
expertise path can reach a health chunk even by mistake -- which is the
property the whole separate store exists for, and the only one that is not
just a naming convention.

Running this is safe on every deploy. It never drops anything.
"""
from __future__ import annotations

import json
import logging
import os
from typing import Any, Optional

logger = logging.getLogger(__name__)

# Quoted rather than interpolated bare, even though these are constants: a
# future edit that makes either configurable would otherwise be an injection
# point in a statement that runs as the database owner.
HEALTH_DB = "healthrag"
HEALTH_ROLE = "healthrag"


def _quote_ident(name: str) -> str:
    if not name.replace("_", "").isalnum():
        raise ValueError(f"refusing to use {name!r} as an identifier")
    return '"' + name.replace('"', '""') + '"'


def _secret(secret_arn: str) -> dict:
    import boto3

    client = boto3.client("secretsmanager")
    return json.loads(client.get_secret_value(SecretId=secret_arn)["SecretString"])


def provision(admin_conn: Any, *, secret_arn: Optional[str] = None) -> dict:
    """Create the database, the role and the grants. Idempotent.

    `admin_conn` is a connection to *another* database on the same instance --
    the expertise one -- because you cannot create a database from inside it.
    """
    secret_arn = secret_arn or os.environ.get("HEALTH_POSTGRES_SECRET_ARN", "")
    if not secret_arn:
        return {"status": "skipped", "reason": "HEALTH_POSTGRES_SECRET_ARN is not set"}

    secret = _secret(secret_arn)
    dbname = secret.get("dbname") or HEALTH_DB
    username = secret.get("username") or HEALTH_ROLE
    password = secret.get("password") or ""
    if not password:
        return {"status": "failed", "reason": "the health secret carries no password"}

    db_ident = _quote_ident(dbname)
    role_ident = _quote_ident(username)

    # CREATE DATABASE cannot run inside a transaction block. The caller's
    # connection is left as it was found.
    #
    # The rollback is not tidiness. psycopg2 refuses to change `autocommit`
    # while a transaction is open, and `db_clients.get_pg_connection` opens one
    # on every call -- its liveness ping is a bare `SELECT 1` with no commit
    # behind it, so the connection this is handed is *always* mid-transaction
    # and the assignment below would raise before a single statement ran.
    admin_conn.rollback()
    previous_autocommit = admin_conn.autocommit
    admin_conn.autocommit = True
    actions: list[str] = []
    try:
        with admin_conn.cursor() as cur:
            # ── the role ────────────────────────────────────────────────────
            cur.execute("SELECT 1 FROM pg_roles WHERE rolname = %s", (username,))
            if cur.fetchone() is None:
                cur.execute(f"CREATE ROLE {role_ident} LOGIN PASSWORD %s", (password,))
                actions.append("role created")
            else:
                # Keeps the role usable after a secret rotation. Without it a
                # rotated secret leaves the store unreachable and the failure
                # looks like a network problem.
                cur.execute(f"ALTER ROLE {role_ident} WITH LOGIN PASSWORD %s", (password,))
                actions.append("role password synced")

            # ── the database ────────────────────────────────────────────────
            cur.execute("SELECT 1 FROM pg_database WHERE datname = %s", (dbname,))
            if cur.fetchone() is None:
                cur.execute(f"CREATE DATABASE {db_ident} OWNER {role_ident}")
                actions.append("database created")
            else:
                actions.append("database already present")

            # ── the grants, in the order that makes them mean something ─────
            #
            # PUBLIC first. Every role is a member of PUBLIC and a new database
            # grants it CONNECT, so revoking from the expertise role while
            # leaving PUBLIC alone isolates nothing at all.
            cur.execute(f"REVOKE ALL ON DATABASE {db_ident} FROM PUBLIC")
            actions.append("PUBLIC revoked")
            cur.execute(f"GRANT CONNECT ON DATABASE {db_ident} TO {role_ident}")
            actions.append("health role granted CONNECT")
    finally:
        admin_conn.autocommit = previous_autocommit

    logger.info("health database provisioned: %s", "; ".join(actions))
    return {"status": "success", "database": dbname, "role": username, "actions": actions}


def verify_isolation(admin_conn: Any, expertise_role: str,
                     dbname: str = HEALTH_DB) -> dict:
    """Ask the database whether the expertise role can reach the health store.

    A grant is the kind of thing that reviews as correct and is not, so this
    asks Postgres rather than trusting the statements above ran in the right
    order. `has_database_privilege` accounts for membership in PUBLIC, which is
    exactly the path a hand-written check would miss.

    What it does **not** prove: this connection is the RDS master user, which
    holds `CREATEROLE` and therefore admin option on the health role it
    created. It can grant itself membership and get in deliberately. The
    isolation here is against the accidental path -- a retrieval query on the
    expertise connection reaching a medical record -- which is the one that
    actually happens. Closing the deliberate path needs the expertise Lambdas
    to run as a non-master role, which is a change to how every function
    connects and is named here rather than half-done.
    """
    with admin_conn.cursor() as cur:
        cur.execute("SELECT 1 FROM pg_database WHERE datname = %s", (dbname,))
        if cur.fetchone() is None:
            return {"isolated": False, "reason": f"no {dbname} database"}
        cur.execute("SELECT has_database_privilege(%s, %s, 'CONNECT')",
                    (expertise_role, dbname))
        can_connect = bool(cur.fetchone()[0])
    return {
        "isolated": not can_connect,
        "expertiseRole": expertise_role,
        "database": dbname,
        "canConnect": can_connect,
        "reason": "" if not can_connect else
                  f"{expertise_role} can still CONNECT to {dbname}",
    }
