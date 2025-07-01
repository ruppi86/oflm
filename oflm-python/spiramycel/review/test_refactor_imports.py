import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]  # spiramycel/
PKG_ROOT = ROOT.parent  # directory containing the 'spiramycel' package
if 'spiramycel' not in sys.modules:
    sys.path.append(str(PKG_ROOT))


print("✅ Path patched – starting import smoke-test")

# 1. Token constants
from spiramycel.token_constants import START_TOKEN, END_TOKEN, PAD_TOKEN
print(f"TOKENS -> START: {START_TOKEN:#04x}, END: {END_TOKEN:#04x}, PAD: {PAD_TOKEN:#04x}")

# 2. Training utils
from spiramycel.training_utils import (
    determine_model_scale_and_folders,
    discover_training_data,
    get_file_size_kb,
)
print("Training_utils imported OK → helper functions available: ",
      [determine_model_scale_and_folders.__name__, discover_training_data.__name__, get_file_size_kb.__name__])

# 3. Logging utils
from spiramycel.logging_utils import setup_experiment_logging, create_condition_logger
main_log, ts = setup_experiment_logging()
print("Main log file created:", main_log)

cond_logger, cond_log_path = create_condition_logger("smoke_test", ts)
cond_logger.info("This is a test line from smoke-test logger.")
print("Condition log file created:", cond_log_path)

print("🎉 All imports & basic calls succeeded – refactor looks healthy.") 