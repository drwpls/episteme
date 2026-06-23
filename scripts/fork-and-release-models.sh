#!/bin/bash
# Script to fork a8nova/kokoro-82m-android and create a release with model files
# Run this script after authenticating with: gh auth login

set -e

echo "=== Forking a8nova/kokoro-82m-android to drwpls ==="
gh repo fork a8nova/kokoro-82m-android --clone=false --remote=false --default-branch-only

echo ""
echo "=== Waiting for fork to be ready ==="
sleep 5

echo ""
echo "=== Creating release v1.0.0 ==="
gh release create v1.0.0 \
  --repo drwpls/kokoro-82m-android \
  --title "Kokoro-82M Android Models v1.0.0" \
  --notes "Forked from a8nova/kokoro-82m-android for Episteme Reader GPU TTS support.

Model files for Adreno GPU-accelerated TTS.

Files included:
- model.bin (Kokoro-82M fp16 weights)
- tokenizer.json
- config.json
- espeak-data/en_dict (English phonemizer)
- voices/af_heart.bin (Default voice pack)

Total size: ~165 MB"

echo ""
echo "=== Release created successfully! ==="
echo ""
echo "Now you need to upload the model files to the release."
echo "Visit: https://github.com/drwpls/kokoro-82m-android/releases/tag/v1.0.0"
echo ""
echo "To upload files manually:"
echo "1. Go to the release page"
echo "2. Click 'Edit'"
echo "3. Drag and drop these files from a8nova's repo:"
echo "   - model.bin"
echo "   - tokenizer.json"
echo "   - config.json"
echo "   - espeak-data/en_dict"
echo "   - voices/af_heart.bin"
echo ""
echo "Or download from HuggingFace first:"
echo "https://huggingface.co/a8nova/kokoro-82m-android/tree/main"
