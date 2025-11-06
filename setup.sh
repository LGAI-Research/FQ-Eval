#!/bin/bash
set -euo pipefail

echo "Setting up FQ-Eval environment..."

# Detect Python version (major.minor)
python_version=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")' 2>/dev/null || true)
if [[ -z "${python_version}" ]]; then
  echo "❌ Could not find python3 on PATH"
  exit 1
fi
echo "Python version: $python_version"

# Require 3.7+
python3 - <<'PY'
import sys
ok = sys.version_info >= (3, 7)
print("✅ Python version is compatible" if ok else f"❌ Warning: Python 3.7+ required, found {sys.version.split()[0]}")
raise SystemExit(0 if ok else 1)
PY

# Install dependencies
echo "Installing required packages..."
python3 -m pip install -r requirements.txt

# Make scripts executable
echo "Making scripts executable..."
chmod +x scoring_auto_eval.sh
chmod +x nway_auto_eval.sh

echo ""
echo "✅ Setup complete!"
echo ""
echo "Next steps:"
echo "1. Set your OpenAI API key:"
echo "   export OPENAI_API_KEY='your-api-key-here'"
echo ""
echo "2. Test installation with sample data:"
echo "   ./scoring_auto_eval.sh --input raw_data/your_data.csv --user_models final_fq --key \$OPENAI_API_KEY --sample_size 5"