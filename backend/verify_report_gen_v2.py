
import sys
import os

# Add backend to path so we can import report_generation
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

print("Checking imports...")
try:
    import llama_cpp
    print("SUCCESS: llama_cpp imported")
except ImportError as e:
    print(f"FAILURE: llama_cpp not found: {e}")

try:
    import weasyprint
    print("SUCCESS: weasyprint imported")
except ImportError as e:
    print(f"FAILURE: weasyprint not found: {e}")

try:
    from report_generation import ReportGenerator
    print("SUCCESS: report_generation imported")
except ImportError as e:
    print(f"FAILURE: report_generation import failed: {e}")

print("Verifiction script finished.")
