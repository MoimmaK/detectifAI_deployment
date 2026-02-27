
import urllib.request
import urllib.error
import os
import sys

print("Python is running")

# Create a dummy video file
with open("test_upload_verify_2.mp4", "wb") as f:
    f.write(b"dummy video content")

url = "http://localhost:5000/api/v2/video/upload"
boundary = "---------------------------123456789012345678901234567"

def build_multipart_data(data_dict, file_info):
    body = []
    # Add fields
    for key, value in data_dict.items():
        body.append(f"--{boundary}".encode())
        body.append(f'Content-Disposition: form-data; name="{key}"'.encode())
        body.append(b"")
        body.append(value.encode())
    
    # Add file
    file_key, filename, file_content = file_info
    body.append(f"--{boundary}".encode())
    body.append(f'Content-Disposition: form-data; name="{file_key}"; filename="{filename}"'.encode())
    body.append(b"Content-Type: video/mp4")
    body.append(b"")
    body.append(file_content)
    
    body.append(f"--{boundary}--".encode())
    body.append(b"")
    return b"\r\n".join(body)

# Test 1: Upload with user_id
print("Test 1: Upload with user_id...")
with open("test_upload_verify_2.mp4", "rb") as f:
    file_content = f.read()

data_body = build_multipart_data(
    {'user_id': 'test_user_verify', 'config_type': 'detectifai'},
    ('video', 'test_upload_verify_2.mp4', file_content)
)

req = urllib.request.Request(url, data=data_body)
req.add_header('Content-Type', f'multipart/form-data; boundary={boundary}')

try:
    with urllib.request.urlopen(req) as response:
        print(f"✅ SUCCESS: Upload accepted. Status: {response.status}")
except urllib.error.HTTPError as e:
    print(f"Status: {e.code}")
    if e.code == 415:
        print("❌ FAILED: Still getting 415 Unsupported Media Type")
    elif e.code in [401, 403]:
        print("✅ SUCCESS: 415 avoided (Got 401/403 which is expected for invalid user)")
    else:
        print(f"✅ SUCCESS: 415 avoided. Got {e.code}")
except Exception as e:
    print(f"❌ FAILED: Connection error or other: {e}")

# Test 2: Upload WITHOUT user_id
print("\nTest 2: Upload WITHOUT user_id...")
data_body = build_multipart_data(
    {'config_type': 'detectifai'},
    ('video', 'test_upload_verify_2.mp4', file_content)
)

req = urllib.request.Request(url, data=data_body)
req.add_header('Content-Type', f'multipart/form-data; boundary={boundary}')

try:
    with urllib.request.urlopen(req) as response:
        print(f"✅ SUCCESS: Upload accepted. Status: {response.status}")
except urllib.error.HTTPError as e:
    print(f"Status: {e.code}")
    if e.code == 415:
        print("❌ FAILED: Still getting 415 Unsupported Media Type")
    elif e.code == 401:
        print("✅ SUCCESS: Correctly updated to 401 (Missing ID), avoided 415.")
    else:
        print(f"✅ SUCCESS: 415 avoided. Got {e.code}")
except Exception as e:
    print(f"❌ FAILED: Connection error or other: {e}")

# Cleanup
try:
    os.remove("test_upload_verify_2.mp4")
except:
    pass
