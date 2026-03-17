"""
scripts/memgraph/connect-test.py

Validates Memgraph connectivity and schema state.
Reads the host from SSM /memgraph/host by default.

Usage:
    python scripts/memgraph/connect-test.py
    python scripts/memgraph/connect-test.py --host <ip>
    python scripts/memgraph/connect-test.py --host <ip> --port 7687

Dependencies:
    pip install neo4j boto3
"""

import argparse
import sys

import boto3
from neo4j import GraphDatabase  # Bolt-compatible with Memgraph


def get_host_from_ssm() -> str:
    ssm = boto3.client("ssm")
    return ssm.get_parameter(Name="/memgraph/host")["Parameter"]["Value"]


def run(host: str, port: int = 7687) -> None:
    uri = f"bolt://{host}:{port}"
    # Memgraph Community has no auth; pass empty strings
    driver = GraphDatabase.driver(uri, auth=("", ""))
    print(f"Connecting → {uri}")

    with driver.session() as s:
        msg = s.run("RETURN 'alive' AS msg").single()["msg"]
        print(f"Ping      : {msg}")

        node_count = s.run("MATCH (n) RETURN count(n) AS c").single()["c"]
        print(f"Nodes     : {node_count}")

        edge_count = s.run("MATCH ()-[r]->() RETURN count(r) AS c").single()["c"]
        print(f"Edges     : {edge_count}")

        print("Constraints:")
        for c in s.run("SHOW CONSTRAINT INFO").data():
            print(f"  {c}")

        print("Indexes:")
        for i in s.run("SHOW INDEX INFO").data():
            print(f"  {i}")

    driver.close()
    print("OK")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate Memgraph connectivity")
    parser.add_argument(
        "--host",
        default=None,
        help="Memgraph host IP or hostname (default: read from SSM /memgraph/host)",
    )
    parser.add_argument("--port", type=int, default=7687, help="Bolt port (default: 7687)")
    args = parser.parse_args()

    host = args.host or get_host_from_ssm()
    try:
        run(host, args.port)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
