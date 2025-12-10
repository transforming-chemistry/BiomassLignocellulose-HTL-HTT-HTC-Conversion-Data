#!/bin/bash
# Sync master_dataset.csv from the main DB directory


SOURCE="../DB/Unified_HTT_Biomass_Database.csv"
TARGET="master_dataset.csv"

if [ ! -f "$SOURCE" ]; then
    echo "❌ Source file not found: $SOURCE"
    exit 1
fi

echo "Syncing master dataset from source..."
cp "$SOURCE" "$TARGET"

ROWS=$(tail -n +2 "$TARGET" | wc -l)
COLS=$(head -1 "$TARGET" | tr ',' '\n' | wc -l)

echo "✅ Updated master_dataset.csv"
echo "   Rows: $ROWS"
echo "   Columns: $COLS"
echo ""
echo "To regenerate metadata.json, run:"
echo "   python generate_metadata.py"
