import sys
import os

print(f"Python Executable: {sys.executable}")
print(f"Python Version: {sys.version}")

try:
    import olga
    print(f"\n[SUCCESS] Imported 'olga'")
    print(f"Location: {os.path.dirname(olga.__file__)}")
except ImportError as e:
    print(f"\n[FAIL] Could not import 'olga': {e}")
    sys.exit(1)

# Inspect Sequence Generation
try:
    import olga.sequence_generation as sg
    print(f"\n[SUCCESS] Imported 'olga.sequence_generation'")
    print(f"File: {sg.__file__}")
    print("--- Available Attributes in sequence_generation ---")
    
    # Filter out internal python attributes like __name__
    attrs = [x for x in dir(sg) if not x.startswith('__')]
    print(attrs)
    
    if 'SequenceGeneration' in attrs:
        print("\n>>> CONFIRMATION: 'SequenceGeneration' class EXISTS.")
    else:
        print("\n>>> CRITICAL FAILURE: 'SequenceGeneration' class MISSING.")
        
except ImportError as e:
    print(f"\n[FAIL] Could not import 'olga.sequence_generation': {e}")

# Inspect Load Model
try:
    import olga.load_model as lm
    print(f"\n[SUCCESS] Imported 'olga.load_model'")
    attrs = [x for x in dir(lm) if not x.startswith('__')]
    print(f"Attributes: {attrs}")
except ImportError as e:
    print(f"\n[FAIL] Could not import 'olga.load_model': {e}")