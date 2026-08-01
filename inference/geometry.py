from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]

sys.path.append(str(ROOT / "training" / "geometry"))

# Temporary values until we wire the geometry engine directly.
# These will be replaced with actual function calls.

class GeometryInference:

    def run(self):

        return {
            "volume": 159.64,
            "mass": 129.30
        }