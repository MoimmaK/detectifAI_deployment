@echo off
echo ================================================================================
echo FIXING VIDEO CAPTIONING IMPORTS
echo ================================================================================

echo.
echo Step 1: Clearing Python cache...
python clear_cache_and_test.py

echo.
echo Step 2: Installing dependencies...
pip install torch torchvision transformers sentence-transformers Pillow numpy pymongo

echo.
echo Step 3: Testing imports...
python test_import.py

echo.
echo ================================================================================
echo FIX COMPLETE!
echo ================================================================================
echo.
echo If imports are successful, you can now run:
echo   python app.py
echo.
pause
