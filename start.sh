#!/bin/bash
# start.sh

cd "$(dirname "$0")"

mkdir -p output input models

echo "Starting ComputerVision Counter Images..."
echo ""

if command -v python3 &> /dev/null; then
    PYTHON_CMD="python3"
elif command -v py &> /dev/null; then
    PYTHON_CMD="py -3"
else
    PYTHON_CMD="python"
fi

if [ ! -d ".venv" ]; then
    echo "[INFO] Creating virtual environment..."
    $PYTHON_CMD -m venv .venv
fi

source .venv/bin/activate
echo "[INFO] Updating dependencies..."
pip install -r requirements.txt --quiet

python src/start_app.py

echo ""
read -n 1 -s -r -p "Press any key to continue..."
echo ""
