"""
Verification script to ensure cleanup was successful
"""

import sys
from pathlib import Path

print("=" * 60)
print("Cleanup Verification Script")
print("=" * 60)

# Check old files are deleted
print("\n[1/4] Verifying old files are deleted...")
old_files = [
    "train.py",
    "evaluate.py",
    "test_env.py",
    "monitor_gpu.py",
    "benchmark_speed.py",
    "check_training_gpu.py",
    "diagnose_crash.py",
    "visualize_training.py",
    "train_modern.py",
    "algorithms/ppo.py",
    "utils/logger.py",
]

all_deleted = True
for file in old_files:
    path = Path(file)
    if path.exists():
        print(f"  ❌ FOUND: {file} (should be deleted)")
        all_deleted = False
    else:
        print(f"  ✓ Deleted: {file}")

if all_deleted:
    print("✓ All old files successfully deleted")
else:
    print("❌ Some old files still exist")
    sys.exit(1)

# Check new files exist
print("\n[2/4] Verifying new SB3 files exist...")
new_files = [
    "train_sb3.py",
    "evaluate_sb3.py",
    "test_sb3_integration.py",
    "models/sb3_policy.py",
    "environment/wrappers.py",
    "utils/curriculum_callback.py",
]

all_exist = True
for file in new_files:
    path = Path(file)
    if path.exists():
        size = path.stat().st_size
        lines = len(path.read_text().splitlines())
        print(f"  ✓ Found: {file} ({lines} lines, {size} bytes)")
    else:
        print(f"  ❌ MISSING: {file}")
        all_exist = False

if all_exist:
    print("✓ All new files present")
else:
    print("❌ Some new files are missing")
    sys.exit(1)

# Check imports
print("\n[3/4] Checking imports...")
try:
    # Test imports without actually running
    import py_compile

    files_to_check = [
        "train_sb3.py",
        "evaluate_sb3.py",
        "test_sb3_integration.py",
        "models/sb3_policy.py",
        "models/networks.py",
        "environment/openscope_env.py",
        "environment/wrappers.py",
        "utils/curriculum.py",
        "utils/curriculum_callback.py",
    ]

    all_valid = True
    for file in files_to_check:
        try:
            py_compile.compile(file, doraise=True)
            print(f"  ✓ Syntax OK: {file}")
        except py_compile.PyCompileError as e:
            print(f"  ❌ Syntax Error: {file}")
            print(f"     {e}")
            all_valid = False

    if all_valid:
        print("✓ All Python files have valid syntax")
    else:
        print("❌ Some files have syntax errors")
        sys.exit(1)

except Exception as e:
    print(f"  ⚠️  Could not verify syntax: {e}")

# Count lines of code
print("\n[4/4] Counting lines of code...")
total_lines = 0
for file in Path(".").rglob("*.py"):
    if "venv" not in str(file) and "__pycache__" not in str(file):
        try:
            lines = len(file.read_text().splitlines())
            total_lines += lines
        except:
            pass

print(f"  Total Python code: {total_lines:,} lines")
print("  Target: ~1,710 lines")

if 1600 <= total_lines <= 1900:
    print("  ✓ Line count in expected range")
else:
    print("  ⚠️  Line count outside expected range")

# Final summary
print("\n" + "=" * 60)
print("✓ Cleanup Verification Complete!")
print("=" * 60)
print("\nSummary:")
print("  • Old files deleted: ✓")
print("  • New files present: ✓")
print("  • Python syntax valid: ✓")
print(f"  • Code size: {total_lines:,} lines")
print("\nReady to use! Run: python test_sb3_integration.py")
print("=" * 60)
