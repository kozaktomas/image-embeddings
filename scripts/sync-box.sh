#!/usr/bin/env bash
# Sync the repository to the GPU build machine. Git stays on the development host;
# the build machine has no GitHub access.
#
# Override the target with BOX_HOST and BOX_DEST.
set -euo pipefail

REMOTE="${BOX_HOST:-box}"
DEST="${BOX_DEST:-\$HOME/dev/image-embeddings}"

DEST="$(ssh "$REMOTE" "eval echo $DEST")"
ssh "$REMOTE" "mkdir -p '$DEST'"
rsync -a --delete \
    --exclude 'venv/' \
    --exclude '.git/' \
    --exclude 'models/' \
    --exclude 'testdata/' \
    --exclude '__pycache__/' \
    --exclude '.pytest_cache/' \
    ./ "$REMOTE:$DEST/"
echo "synced -> $REMOTE:$DEST"
