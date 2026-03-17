#!/bin/bash
# scripts/memgraph/userdata.sh
#
# EC2 UserData bootstrap for Memgraph Community on Ubuntu 24.04 ARM64.
# This file is the standalone reference copy.  The CloudFormation template
# (cfn/memgraph-ec2.yaml) embeds an inline version via Fn::Base64 + Fn::Sub so
# that ${AWS::StackName} and ${AWS::Region} resolve at deploy time.
#
# Do NOT run this file directly against an EC2 instance; use the CFN template.
set -euxo pipefail

# ── 1. System update ──────────────────────────────────────────────────────────
apt-get update -y && apt-get upgrade -y

# ── 2. Install Docker (official repo, ARM64-safe) ─────────────────────────────
apt-get install -y ca-certificates curl gnupg lsb-release
install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg \
  | gpg --dearmor -o /etc/apt/keyrings/docker.gpg
chmod a+r /etc/apt/keyrings/docker.gpg
echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] \
  https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable" \
  > /etc/apt/sources.list.d/docker.list
apt-get update -y
apt-get install -y docker-ce docker-ce-cli containerd.io docker-compose-plugin

systemctl enable docker && systemctl start docker
usermod -aG docker ubuntu

# ── 3. Write docker-compose.yml ───────────────────────────────────────────────
# t4g.micro has 1 GB RAM; mem_limit=700m is intentional – do not raise it.
mkdir -p /opt/memgraph
cat > /opt/memgraph/docker-compose.yml << 'COMPOSE'
version: "3.8"
services:
  memgraph:
    image: memgraph/memgraph-platform:latest
    container_name: memgraph
    restart: always
    ports:
      - "7687:7687"
      - "7444:7444"
    volumes:
      - mg_data:/var/lib/memgraph
      - mg_log:/var/log/memgraph
      - mg_etc:/etc/memgraph
    environment:
      - MEMGRAPH_TELEMETRY_ENABLED=false
    command: >
      --log-level=WARNING
      --storage-snapshot-interval-sec=300
      --storage-wal-enabled=true
      --storage-snapshot-on-exit=true
      --storage-recover-on-startup=true
    mem_limit: 700m
    memswap_limit: 700m
volumes:
  mg_data:
  mg_log:
  mg_etc:
COMPOSE

# ── 4. Start Memgraph ─────────────────────────────────────────────────────────
cd /opt/memgraph && docker compose up -d

# ── 5. Wait for Bolt to be ready (up to 30 × 5 s = 150 s) ───────────────────
for i in $(seq 1 30); do
  if docker exec memgraph mgconsole --no-history \
       --execute "RETURN 1;" > /dev/null 2>&1; then
    echo "Memgraph ready after ${i} attempts."
    break
  fi
  sleep 5
done

# ── 6. Signal CloudFormation ──────────────────────────────────────────────────
# cfn-bootstrap is not pre-installed on Ubuntu 24.04; install it first.
# In the CFN template, ${AWS::StackName} and ${AWS::Region} are substituted
# by Fn::Sub before this script runs on the instance.
pip3 install \
  https://s3.amazonaws.com/cloudformation-examples/aws-cfn-bootstrap-py3-latest.tar.gz \
  --break-system-packages
/usr/local/bin/cfn-signal \
  --exit-code $? \
  --stack "${AWS::StackName}" \
  --resource MemgraphEC2Instance \
  --region "${AWS::Region}"
