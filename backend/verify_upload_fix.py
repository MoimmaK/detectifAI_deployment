
import requests
import os

# Create a dummy video file
with open("test_upload_verify.mp4", "wb") as f:
    f.write(b"dummy video content")

url = "http://localhost:5000/api/v2/video/upload"

# Test 1: Upload with user_id in form data (Should succeed or fail with something other than 415)
print("Test 1: Upload with user_id in form data...")
files = {'video': ('test_upload_verify.mp4', open('test_upload_verify.mp4', 'rb'), 'video/mp4')}
data = {'user_id': 'test_user_verify', 'config_type': 'detectifai'}

try:
    response = requests.post(url, files=files, data=data)
    print(f"Status: {response.status_code}")
    print(f"Response: {response.text}")
    
    if response.status_code == 415:
        print("❌ FAILED: Still getting 415 Unsupported Media Type")
    elif response.status_code in [200, 201]:
        print("✅ SUCCESS: Upload accepted")
    elif response.status_code == 401:
        print("✅ PARTIAL SUCCESS: 415 avoided, got 401 (Expected if test user invalid/middleware strict)")
    else:
        print(f"✅ SUCCESS: 415 avoided. Got {response.status_code}")

except requests.exceptions.ConnectionError:
    print("❌ FAILED: Could not connect to server. Is it running?")

# Test 2: Upload WITHOUT user_id (Should return 401, NOT 415)
print("\nTest 2: Upload WITHOUT user_id...")
files = {'video': ('test_upload_verify.mp4', open('test_upload_verify.mp4', 'rb'), 'video/mp4')}
# No user_id in data

try:
    response = requests.post(url, files=files, data={})
    print(f"Status: {response.status_code}")
    
    if response.status_code == 415:
        print("❌ FAILED: Still getting 415 Unsupported Media Type")
    elif response.status_code == 401:
        print("✅ SUCCESS: Correctly identified missing user_id (401), avoided 415.")
    else:
         print(f"⚠️ Unexpected status: {response.status_code}")

except Exception as e:
    print(f"Error: {e}")

# Cleanup
try:
    os.remove("test_upload_verify.mp4")
except:
    pass
