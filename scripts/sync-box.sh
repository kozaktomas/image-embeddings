#!/usr/bin/env bash
# Synchronizuje repo z Pi na box. Git zustava na Pi, box nema pristup na GitHub.
set -euo pipefail

REMOTE="${BOX_HOST:-box}"
DEST="${BOX_DEST:-/home/panbotka/dev/image-embeddings}"

ssh "$REMOTE" "mkdir -p '$DEST'"
rsync -a --delete \
    --exclude 'venv/' \
    --exclude '.git/' \
    --exclude 'models/' \
    --exclude 'testdata/' \
    --exclude '__pycache__/' \
    --exclude '.pytest_cache/' \
    ./ "$REMOTE:$DEST/"
echo "synchronizovano -> $REMOTE:$DEST"
