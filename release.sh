#!/usr/bin/env bash
# Bump the version, commit, and tag — pushing the tag triggers the release CI.
# Usage: ./release.sh [major|minor|patch]   (default: patch)
set -euo pipefail
cd "$(dirname "$0")"

kind="${1:-patch}"

if [[ -n "$(git status --porcelain)" ]]; then
    echo "error: working tree is not clean — commit or stash first" >&2
    exit 1
fi

current=$(grep -m1 '^version = ' Cargo.toml | sed 's/version = "\(.*\)"/\1/')
IFS=. read -r maj min pat <<< "$current"
case "$kind" in
    major) maj=$((maj + 1)); min=0; pat=0 ;;
    minor) min=$((min + 1)); pat=0 ;;
    patch) pat=$((pat + 1)) ;;
    *) echo "usage: $0 [major|minor|patch]" >&2; exit 1 ;;
esac
new="$maj.$min.$pat"

echo "bumping $current -> $new"
sed -i "0,/^version = \"$current\"/s//version = \"$new\"/" Cargo.toml
cargo check --quiet   # refresh Cargo.lock

git add Cargo.toml Cargo.lock
git commit -m "release v$new"
git tag "v$new"

echo "tagged v$new"
read -r -p "push now (git push && git push --tags)? [y/N] " answer || answer=""
if [[ "$answer" == [yY]* ]]; then
    git push && git push --tags
    echo "pushed — the release workflow is building on GitHub"
else
    echo "not pushed — run: git push && git push --tags"
fi
