
import sys
import os

# Add backend to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

print("Verifying backend imports...")
try:
    from app import app, REPORT_GENERATION_AVAILABLE, generate_report
    print("SUCCESS: app imported")
    
    if REPORT_GENERATION_AVAILABLE:
        print("SUCCESS: REPORT_GENERATION_AVAILABLE is True")
    else:
        print("WARNING: REPORT_GENERATION_AVAILABLE is False (Dependencies missing or Model missing?)")
        
    print("Verification complete.")
except ImportError as e:
    print(f"FAILURE: Import failed: {e}")
except Exception as e:
    print(f"FAILURE: Exception: {e}")
