#!/usr/bin/env bash
# ---------------------------------------------------------------------------
#  PI Detection - start the Pipeline Wizard (Linux / Raspberry Pi)
#
#  Activates the project virtual environment (venv/) and launches the UI.
#  Run it from any directory:  ./start_ui_linux.sh     (or: bash start_ui_linux.sh)
#
#  Extra arguments are forwarded to the target, e.g.  ./start_ui_linux.sh --help
#  Set UI_TARGET to run something else with the venv active, e.g.
#      UI_TARGET=train.py ./start_ui_linux.sh
# ---------------------------------------------------------------------------

# Run from the folder this script lives in, whatever the current directory is.
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd) || exit 1
cd "$SCRIPT_DIR" || exit 1

UI_TARGET=${UI_TARGET:-pipeline_ui.py}
VENV_ACT="venv/bin/activate"
VENV_PY="venv/bin/python"

if [ ! -f "$VENV_ACT" ]; then
    echo
    echo "[ERROR] Virtual environment not found (expected $VENV_ACT)."
    echo
    echo "        Create it once from this folder:"
    echo "            python3 -m venv venv"
    echo "            source venv/bin/activate"
    echo "            pip install -r requirements.txt"
    echo "            pip install RPi.GPIO        # Pi only"
    echo
    printf "Press Enter to close..."
    read -r _
    exit 1
fi

echo "Activating virtual environment ..."
# shellcheck disable=SC1090
. "$VENV_ACT"

# Prefer the activated interpreter; fall back to the venv python directly.
if command -v python >/dev/null 2>&1; then
    PY=python
else
    PY="./$VENV_PY"
fi

echo "Starting $UI_TARGET ..."
"$PY" "$UI_TARGET" "$@"
rc=$?

if [ "$rc" -ne 0 ]; then
    echo
    echo "[$UI_TARGET exited with code $rc]"
    printf "Press Enter to close..."
    read -r _
fi
exit "$rc"
