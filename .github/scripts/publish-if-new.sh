#!/usr/bin/env bash
#
# Idempotent crate publisher used by the Release workflow.
#
# Usage: publish-if-new.sh <crate-name> <expected-version>
#
# - Refuses to run if the local Cargo.toml's resolved version doesn't
#   match the expected one (catches stale checkouts / bad inputs).
# - Queries crates.io for the latest version of the crate.
# - If the expected version is already published, the script logs and
#   exits 0 (idempotent — re-runs of a partial release don't fail).
# - Otherwise runs `cargo publish -p <crate>` with the configured
#   CARGO_REGISTRY_TOKEN, then waits for the index to confirm the
#   upload before returning.

set -euo pipefail

if [ "$#" -ne 2 ]; then
  echo "usage: publish-if-new.sh <crate> <version>" >&2
  exit 64
fi

CRATE="$1"
VERSION="$2"

# Validate the inputs (the workflow already does this for VERSION, but
# double-checking keeps this script safe to run by hand).
if ! [[ "$CRATE" =~ ^[a-z][a-z0-9_-]*$ ]]; then
  echo "::error::invalid crate name: $CRATE" >&2
  exit 65
fi
if ! [[ "$VERSION" =~ ^[0-9]+\.[0-9]+\.[0-9]+(-[A-Za-z0-9.]+)?$ ]]; then
  echo "::error::invalid version: $VERSION" >&2
  exit 65
fi

echo "::group::checking $CRATE@$VERSION"

# 1. Confirm the manifest version actually matches what we're trying to ship.
LOCAL_VERSION=$(cargo pkgid -p "$CRATE" 2>/dev/null | sed -E 's/.*[#@]([0-9]+\.[0-9]+\.[0-9]+(-[A-Za-z0-9.]+)?)$/\1/')
if [ "$LOCAL_VERSION" != "$VERSION" ]; then
  echo "::error::$CRATE manifest version is $LOCAL_VERSION, expected $VERSION"
  echo "::endgroup::"
  exit 1
fi

# 2. Ask crates.io whether this version is already up.
PUBLISHED=$(curl --silent --fail --max-time 30 \
  "https://crates.io/api/v1/crates/$CRATE/$VERSION" \
  | python3 -c 'import json,sys; sys.stdout.write(json.load(sys.stdin).get("version", {}).get("num", ""))' 2>/dev/null \
  || true)

if [ "$PUBLISHED" = "$VERSION" ]; then
  echo "$CRATE@$VERSION is already on crates.io — skipping (idempotent re-run)"
  echo "::endgroup::"
  exit 0
fi

echo "::endgroup::"
echo "::group::publishing $CRATE@$VERSION"

# 3. Publish. cargo publish blocks until the new version is queryable on
#    the index, so the next dependent step in the workflow can resolve
#    it without manual sleep.
cargo publish -p "$CRATE"

echo "::endgroup::"
