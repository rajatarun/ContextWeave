#!/usr/bin/env bash
set -euo pipefail

FUNC_NAME="${FUNC_NAME:-rag-backend}"
BUILD_DIR="${BUILD_DIR:-build}"
ZIP_NAME="${ZIP_NAME:-function.zip}"

REGION="${AWS_REGION:-us-east-1}"
GUARDRAIL_ID="${GUARDRAIL_ID:-}"
GUARDRAIL_VERSION="${GUARDRAIL_VERSION:-1}"

rm -rf "$BUILD_DIR"
mkdir -p "$BUILD_DIR"

cp -r src "$BUILD_DIR/src"
cp package.json "$BUILD_DIR/package.json"
mkdir -p "$BUILD_DIR/scripts"
cp -r scripts/*.json "$BUILD_DIR/scripts/"

curl -L -o "$BUILD_DIR/rds-ca-bundle.pem" https://truststore.pki.rds.amazonaws.com/global/global-bundle.pem

pushd "$BUILD_DIR" >/dev/null
npm install --omit=dev
popd >/dev/null

rm -f "$ZIP_NAME"
(cd "$BUILD_DIR" && zip -r "../$ZIP_NAME" src package.json rds-ca-bundle.pem node_modules scripts)

aws lambda update-function-code --function-name "$FUNC_NAME" --zip-file "fileb://$ZIP_NAME" --region "$REGION"
aws lambda publish-version --function-name "$FUNC_NAME" --region "$REGION" >/dev/null

if [ -n "$GUARDRAIL_ID" ]; then
  aws lambda update-function-configuration     --function-name "$FUNC_NAME"     --environment "Variables={GUARDRAIL_ID=$GUARDRAIL_ID,GUARDRAIL_VERSION=$GUARDRAIL_VERSION}"     --region "$REGION" >/dev/null
  echo "OK (lambda updated + guardrail env set)"
else
  echo "OK (lambda updated)"
  echo "To create a guardrail: ./guardrail.sh"
  echo "Then set env and re-run deploy:"
  echo "  export GUARDRAIL_ID=<guardrailId>"
  echo "  export GUARDRAIL_VERSION=1"
  echo "  ./deploy.sh"
fi
