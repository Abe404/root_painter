#!/bin/bash

set -e  # Exit on error

# Default to TestPyPI
REPO="testpypi"

# Allow 'prod' as an argument to publish to the real PyPI
if [[ $1 == "prod" ]]; then
    REPO="pypi"
    echo "⚠️  WARNING: You are about to publish to PRODUCTION PyPI! ⚠️"
    read -p "Are you sure? Type 'yes' to continue: " CONFIRM
    if [[ "$CONFIRM" != "yes" ]]; then
        echo "❌ Aborted."
        exit 1
    fi
fi

echo "📦 Building package..."
rm -rf dist/*
python -m build --sdist

echo "🚀 Uploading to $REPO..."
twine upload --repository "$REPO" dist/* --verbose
