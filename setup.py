import subprocess
import sys
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR  = os.path.join(BASE_DIR, 'src')

scripts = [
    "generate_data.py",
    "feature_engineering.py",
    "preprocessing.py",
    "train.py",
    "evaluate.py"
]

print("=" * 40)
print("FlowScore Setup — Running all scripts")
print("=" * 40)

for script in scripts:
    path = os.path.join(SRC_DIR, script)
    print(f"\n▶ Running {script}...")
    result = subprocess.run([sys.executable, path], capture_output=False)
    if result.returncode != 0:
        print(f"\n❌ {script} failed. Stopping.")
        sys.exit(1)
    print(f"✅ {script} done")

print("\n" + "=" * 40)
print("✅ All scripts complete! Run: py flowscore.py")
print("=" * 40)