# tests/conftest.py
import sys
from pathlib import Path

# Add project/src to the path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))