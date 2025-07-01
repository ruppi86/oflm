from pathlib import Path
import sys
import json
from collections import Counter

# Ensure package root importable
ROOT = Path(__file__).resolve().parents[1]  # spiramycel/
PKG_ROOT = ROOT.parent  # oflm-python directory
if str(PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(PKG_ROOT))

from spiramycel.token_constants import START_TOKEN, END_TOKEN, PAD_TOKEN
from spiramycel.training_utils import set_deterministic

# Ecological data generator lives in ecological_data_generator.py
try:
    from spiramycel.ecological_data_generator import EcologicalDataGenerator  # type: ignore
except ImportError:
    # Fallback for standalone run where spiramycel isn't a package install
    from ecological_data_generator import EcologicalDataGenerator  # type: ignore

set_deterministic(42)

def validate_dataset(file_path: Path):
    """Validate JSONL dataset file for field completeness and glyph value range."""
    issues = []
    glyph_counts = Counter()
    total_examples = 0
    silence_tokens = {START_TOKEN, END_TOKEN, PAD_TOKEN}
    total_silence = 0
    total_tokens = 0

    with file_path.open("r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError as e:
                issues.append((line_num, f"JSON error: {e}"))
                continue

            # Basic field presence
            for field in ("conditions", "repair_action"):
                if field not in data:
                    issues.append((line_num, f"Missing field '{field}'"))

            glyph_seq = data.get("repair_action", {}).get("glyph_sequence", [])
            if not glyph_seq:
                issues.append((line_num, "Empty glyph_sequence"))
            else:
                for g in glyph_seq:
                    if not isinstance(g, int):
                        issues.append((line_num, f"Non-int glyph {g}"))
                    elif g < 0x00 or g > PAD_TOKEN:
                        issues.append((line_num, f"Glyph out of range {g:#04x}"))
                    glyph_counts[g] += 1
                    total_tokens += 1
                    if g in silence_tokens:
                        total_silence += 1
            total_examples += 1

    sil_ratio = total_silence / total_tokens if total_tokens else 0.0
    return issues, total_examples, sil_ratio, glyph_counts


def main():
    gen = EcologicalDataGenerator(random_seed=42)
    # Save under training_scenarios so generator doesn't fail
    out_dir = PKG_ROOT / "spiramycel" / "training_scenarios"
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp_file = out_dir / "test_ecological.jsonl"
    gen.generate_training_dataset( num_echoes=500, output_file=str(tmp_file), chaos_mode=False)

    issues, n, sil_ratio, counts = validate_dataset(tmp_file)
    print(f"Validated {n} examples → issues: {len(issues)}; silence ratio {sil_ratio:.2%}")
    print("Most common glyphs:", counts.most_common(5))
    if issues:
        print("Sample issues (first 10):")
        for ln, msg in issues[:10]:
            print(f"  line {ln}: {msg}")

if __name__ == "__main__":
    main() 