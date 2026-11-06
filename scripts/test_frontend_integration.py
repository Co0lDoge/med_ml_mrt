"""
Test end-to-end frontend integration via Next.js proxy on port 3005.
"""
import requests
import json
import base64
import time
import os
import subprocess
import uuid

def get_proxy_base():
    # If in WSL, try gateway IP, otherwise localhost
    try:
        out = subprocess.check_output("ip route show default", shell=True).decode()
        ip = out.split()[2]
        return f"http://{ip}:3005/api/v1"
    except Exception:
        return "http://localhost:3005/api/v1"

PROXY_URL = get_proxy_base()
TEST_IMG = r"D:\Study\Projects\med_ml_mrt\_test_data\patient_T1.tif"
if not os.path.exists(TEST_IMG):
    TEST_IMG = "/mnt/d/Study/Projects/med_ml_mrt/_test_data/patient_T1.tif"

def main():
    email = f"doctor_{int(time.time())}@hospital.test"
    password = "Password123!"

    print(f"[1] Registering doctor {email} via {PROXY_URL}/reg/doctor...")
    r = requests.post(f"{PROXY_URL}/reg/doctor", json={
        "email": email,
        "password": password,
        "fullname": "Dr. Gregory House",
        "org": "Hospital",
        "job": "Radiologist",
        "description": "Neuro"
    })
    print(f"    Register status: {r.status_code}")

    print(f"[2] Logging in via {PROXY_URL}/login...")
    r = requests.post(f"{PROXY_URL}/login", json={"email": email, "password": password})
    assert r.status_code == 200, f"Login failed: {r.text}"
    token = r.json()["access_token"]
    payload = json.loads(base64.b64decode(token.split(".")[1] + "==").decode())
    doctor_id = payload.get("id") or payload.get("sub")
    print(f"    Doctor ID from JWT: {doctor_id}")

    headers = {"Authorization": f"Bearer {token}"}

    print(f"[3] Fetching doctor studies via {PROXY_URL}/mris/author/{doctor_id}...")
    r = requests.get(f"{PROXY_URL}/mris/author/{doctor_id}", headers=headers)
    if r.status_code == 200:
        studies = r.json()
    else:
        # Backend returns not found when author has 0 studies
        studies = []
    print(f"    Initial studies: {len(studies)}")

    print(f"[4] Uploading MRI TIFF via {PROXY_URL}/mri...")
    patient_id = str(uuid.uuid4())
    with open(TEST_IMG, "rb") as f:
        r = requests.post(
            f"{PROXY_URL}/mri",
            headers=headers,
            data={"projection": "cross", "device_id": 1, "external_id": patient_id},
            files={"file": ("patient_T1.tif", f, "image/tiff")}
        )
    assert r.status_code == 200, f"Upload failed: {r.text}"
    mri_id = r.json()["id"]
    print(f"    Uploaded MRI ID: {mri_id}")

    print(f"[5] Waiting for GPU processing by ml_service...")
    for i in range(30):
        time.sleep(2)
        r = requests.get(f"{PROXY_URL}/mri/{mri_id}/nodes", headers=headers)
        if r.status_code == 200 and len(r.json()) > 0:
            nodes = r.json()
            print(f"    [OK] Node created after {(i+1)*2}s: {nodes[0]}")
            break
    else:
        print("    [!] Timed out waiting for node")

    print(f"[6] Verifying image slices via {PROXY_URL}/mri/{mri_id}/images...")
    r = requests.get(f"{PROXY_URL}/mri/{mri_id}/images", headers=headers)
    assert r.status_code == 200, f"Get images failed: {r.text}"
    images = r.json()
    print(f"    Total slices found: {len(images)}")
    assert len(images) > 0, "No slices returned"

    # Test downloading first slice
    first_img_id = images[0]["id"]
    print(f"[7] Testing slice download via {PROXY_URL}/download/{mri_id}/{first_img_id}...")
    r = requests.get(f"{PROXY_URL}/download/{mri_id}/{first_img_id}", headers=headers)
    assert r.status_code == 200, f"Download failed: {r.status_code}"
    print(f"    Downloaded image bytes: {len(r.content)} (PNG header: {r.content[:4] == b'\\x89PNG'})")

    print(f"[8] Verifying segments via {PROXY_URL}/mri/nodes/{nodes[0]['id']}/segments...")
    r = requests.get(f"{PROXY_URL}/mri/nodes/{nodes[0]['id']}/segments", headers=headers)
    assert r.status_code == 200, f"Get segments failed: {r.text}"
    segments = r.json()
    print(f"    Total segmented slice contours: {len(segments)}")
    if len(segments) > 0:
        first_seg = segments[0]
        print(f"    Sample segment contor points count: {len(first_seg.get('contor', []))}")

    print("\n================================================================")
    print("ALL FRONTEND INTEGRATION & PROXY ACCEPTANCE CRITERIA PASSED!")
    print("================================================================")

if __name__ == "__main__":
    main()
