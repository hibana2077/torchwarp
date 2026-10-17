import sys
from pathlib import Path

# make `from golden import golden` work from the test modules
sys.path.insert(0, str(Path(__file__).resolve().parent))
