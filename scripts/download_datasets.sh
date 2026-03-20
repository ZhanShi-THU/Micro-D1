#!/usr/bin/env bash
set -e

# ============================================================
# Dataset Download Script
# Usage: ./download_datasets.sh --data-dir /path/to/data
# ============================================================

DATA_DIR="/data1"

LLava_PRETRAIN_DIR="$DATA_DIR/LLaVA-Pretrain"
COCO_DIR="$DATA_DIR/coco"
VG_DIR="$DATA_DIR/visual_genome"

echo "============================================================"
echo "Dataset Download Script"
echo "============================================================"
echo "Data directory: $DATA_DIR"
echo

# Check ModelScope
if ! python3 -c "import modelscope" 2>/dev/null; then
    echo "ModelScope not found. Installing..."
    pip install modelscope
fi

# ----------------------------------------------------------
# 1. LLaVA-Pretrain 558K (from ModelScope)
# ----------------------------------------------------------
download_llava_pretrain() {
    echo
    echo "============================================================"
    echo "Downloading LLaVA-Pretrain 558K from ModelScope..."
    echo "============================================================"

    if [ -d "$LLava_PRETRAIN_DIR" ] && [ "$(ls -A "$LLava_PRETRAIN_DIR")" ]; then
        echo "LLaVA-Pretrain already exists at $LLava_PRETRAIN_DIR, skipping..."
        return
    fi

    mkdir -p "$LLava_PRETRAIN_DIR"

    echo "Downloading via ModelScope SDK..."
    python3 -c "
from modelscope.msdatasets import MsDataset
ds = MsDataset.load('AI-ModelScope/LLaVA-Pretrain', cache_dir='$LLava_PRETRAIN_DIR')
print('Downloaded successfully')
"
    echo "Done: $LLava_PRETRAIN_DIR"
}

# ----------------------------------------------------------
# 2. COCO Captions (2014 version, from ModelScope)
# ----------------------------------------------------------
download_coco() {
    echo
    echo "============================================================"
    echo "Downloading COCO Captions (2014) from ModelScope..."
    echo "============================================================"

    if [ -d "$COCO_DIR" ] && [ "$(ls -A "$COCO_DIR")" ]; then
        echo "COCO already exists at $COCO_DIR, skipping..."
        return
    fi

    mkdir -p "$COCO_DIR"

    echo "Downloading via ModelScope SDK..."
    python3 -c "
from modelscope.msdatasets import MsDataset

# Download train set
print('Downloading train split...')
ds_train = MsDataset.load('modelscope/coco_2014_caption', subset_name='train', split='train', cache_dir='$COCO_DIR')
print(f'Train: {len(ds_train)} samples')

# Download validation set
print('Downloading validation split...')
ds_val = MsDataset.load('modelscope/coco_2014_caption', subset_name='valid', split='validation', cache_dir='$COCO_DIR')
print(f'Validation: {len(ds_val)} samples')
"
    echo "Done: $COCO_DIR"
    echo "Note: Images are accessed remotely via ModelScope, not stored locally."
}

# ----------------------------------------------------------
# 3. Visual Genome
# ----------------------------------------------------------
download_visual_genome() {
    echo
    echo "============================================================"
    echo "Downloading Visual Genome..."
    echo "============================================================"

    if [ -d "$VG_DIR" ] && [ "$(ls -A "$VG_DIR")" ]; then
        echo "Visual Genome already exists at $VG_DIR, skipping..."
        return
    fi

    mkdir -p "$VG_DIR"
    cd "$VG_DIR"

    # Download from Stanford servers
    echo "Downloading images (part 1, ~50GB)..."
    wget -c https://cs.stanford.edu/people/rak248/VG_100K_2/images.zip
    echo "Downloading images (part 2, ~50GB)..."
    wget -c https://cs.stanford.edu/people/rak248/VG_100K_2/images2.zip

    # Download annotations
    echo "Downloading annotations..."
    wget -c https://cs.stanford.edu/people/rak248/VG_100K_2/annotations.json

    # Note: Full VG dataset with region descriptions requires request
    echo "Note: For full dataset with all region descriptions, visit: https://visualgenome.org/"

    # Unzip images
    echo "Unzipping..."
    mkdir -p images
    unzip -q images.zip -d images/ || true
    unzip -q images2.zip -d images/ || true

    rm -f images.zip images2.zip

    echo "Done: $VG_DIR"
}

# ----------------------------------------------------------
# Parse arguments
# ----------------------------------------------------------
while [[ $# -gt 0 ]]; do
    case $1 in
        --data-dir)
            DATA_DIR="$2"
            shift 2
            ;;
        --llava-only)
            download_llava_pretrain
            exit 0
            ;;
        --coco-only)
            download_coco
            exit 0
            ;;
        --vg-only)
            download_visual_genome
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

mkdir -p "$DATA_DIR"
mkdir -p "$DATA_DIR/LLaVA-Pretrain"
mkdir -p "$DATA_DIR/coco/images" "$DATA_DIR/coco/annotations"
mkdir -p "$DATA_DIR/visual_genome"

# Download all
download_llava_pretrain
download_coco
download_visual_genome

echo
echo "============================================================"
echo "All datasets downloaded to: $DATA_DIR"
echo "============================================================"
