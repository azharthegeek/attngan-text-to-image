#!/usr/bin/env bash
# Downloads preprocessed CUB-200-2011 text annotations + train/test split pickles
# from the DF-GAN repo mirror (CVPR 2022) — same format as AttnGAN/StackGAN.
#
# What this downloads (~50 MB zip):
#   data/birds/text/           — 11,788 .txt files, 10 captions each
#   data/birds/train/filenames.pickle
#   data/birds/test/filenames.pickle
#   data/birds/class_info.pickle   (optional, but included)
#
# Alternative (Kaggle, requires kaggle CLI + free account):
#   kaggle datasets download -d somthirthabhowmk2001/text-to-image-cub-200-2011
#   unzip text-to-image-cub-200-2011.zip -d data/birds/

set -e

DRIVE_ID="1I6ybkR7L64K8hZOraEZDuHh0cCJw5OUj"
OUT_ZIP="birds_metadata.zip"
DATA_DIR="data/birds"

# Ensure we run from the project root
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR/.."

echo "==> Checking for gdown..."
if ! python -c "import gdown" 2>/dev/null; then
    echo "    Installing gdown..."
    pip install -q gdown
fi

echo "==> Downloading birds metadata from Google Drive (ID: $DRIVE_ID)..."
python -m gdown "https://drive.google.com/uc?id=${DRIVE_ID}" -O "$OUT_ZIP"

echo "==> Extracting to $DATA_DIR/ ..."
mkdir -p "$DATA_DIR"
unzip -q "$OUT_ZIP" -d "$DATA_DIR"
rm -f "$OUT_ZIP"

echo "==> Verifying..."
MISSING=0
for f in \
    "$DATA_DIR/text" \
    "$DATA_DIR/train/filenames.pickle" \
    "$DATA_DIR/test/filenames.pickle"; do
    if [ ! -e "$f" ]; then
        echo "    MISSING: $f"
        MISSING=1
    else
        echo "    OK: $f"
    fi
done

if [ "$MISSING" -eq 1 ]; then
    echo ""
    echo "Some files are missing. The zip structure may differ — check what was"
    echo "extracted inside $DATA_DIR/ and move files to the expected paths."
    echo ""
    echo "If the Google Drive link is unavailable, use the Kaggle alternative:"
    echo "  kaggle datasets download -d somthirthabhowmk2001/text-to-image-cub-200-2011"
    echo "  unzip text-to-image-cub-200-2011.zip -d $DATA_DIR/"
    echo ""
    echo "Or generate only the split pickles from existing images (text/ still needed):"
    echo "  python scripts/generate_pickles.py"
    exit 1
fi

echo ""
echo "Done. Run a quick smoke test:"
echo "  python -c \""
echo "  from code.datasets import CUBDataset"
echo "  ds = CUBDataset('data/birds', split='train')"
echo "  print(f'Train: {len(ds)} samples')\""
