#!/bin/bash
set -e
cd "$(dirname "$0")"
echo "Diabetes Disease Prediction - Mac setup"
if ! command -v python3 >/dev/null 2>&1; then
  echo "Python 3 was not found. Install Python from https://www.python.org/downloads/macos/ and try again."
  read -r -p "Press Enter to close..."
  exit 1
fi
if [ ! -d ".venv" ]; then
  python3 -m venv .venv
fi
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python train_model.py
python -m streamlit run app.py
