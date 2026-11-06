"""
End-to-End MRI Processing Test Script
Tests the full lifecycle:
  1. Register a test doctor
  2. Authenticate to obtain a JWT Bearer token
  3. Upload `_test_data/patient_T1.tif` to the Composition API
  4. Poll PostgreSQL (examdb) until nnUNet completes segmentation and persists mask/nodes
"""

import os
import sys
import time
import uuid
import requests
import subprocess

BASE_URL = "http://localhost:8080/api/v1"
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEST_IMAGE = os.path.join(ROOT_DIR, "_test_data", "patient_T1.tif")

# Handle WSL vs Windows paths
if not os.path.exists(TEST_IMAGE):
    # Try WSL mounted path if run from Linux
    wsl_alt = "/mnt/d/Study/Projects/med_ml_mrt/_test_data/patient_T1.tif"
    if os.path.exists(wsl_alt):
        TEST_IMAGE = wsl_alt
    else:
        print(f"[ERROR] Test image not found at: {TEST_IMAGE}")
        sys.exit(1)


def main():
    email = f"doctor_{int(time.time())}@hospital.test"
    password = "Password123!"

    print("================================================================")
    print("           MED_ML_MRT - MRI PROCESSING TEST PIPELINE            ")
    print("================================================================")

    # 1. Register
    print(f"\n[1/4] Registering test doctor ({email})...")
    reg_data = {
        "email": email,
        "password": password,
        "fullname": "Dr. Gregory House",
        "org": "Princeton-Plainsboro Hospital",
        "job": "Diagnostic Medicine",
        "description": "MRI Specialist"
    }
    r = requests.post(f"{BASE_URL}/reg/doctor", json=reg_data)
    if r.status_code != 200:
        print(f"[!] Registration failed (HTTP {r.status_code}): {r.text}")
        sys.exit(1)
    doctor_id = r.json().get("id")
    print(f" [OK] Doctor registered. ID: {doctor_id}")

    # 2. Login
    print("\n[2/4] Logging in...")
    login_data = {
        "email": email,
        "password": password
    }
    r = requests.post(f"{BASE_URL}/login", json=login_data)
    if r.status_code != 200:
        print(f"[!] Login failed (HTTP {r.status_code}): {r.text}")
        sys.exit(1)
    tokens = r.json()
    token = tokens.get("access_token")
    print(" [OK] Access token acquired.")

    # 3. Upload MRI Scan
    patient_id = str(uuid.uuid4())
    print(f"\n[3/4] Uploading scan: {TEST_IMAGE}")
    print(f"      External Patient UUID: {patient_id}")
    print(f"      File size: {os.path.getsize(TEST_IMAGE) / 1024 / 1024:.2f} MB")

    headers = {"Authorization": f"Bearer {token}"}
    with open(TEST_IMAGE, "rb") as f:
        files = {
            "file": ("patient_T1.tif", f, "image/tiff")
        }
        data = {
            "projection": "cross",
            "external_id": patient_id,
            "device_id": 1,
            "description": "Automated e2e test with patient_T1.tif"
        }
        r = requests.post(f"{BASE_URL}/mri", headers=headers, files=files, data=data)

    if r.status_code != 200:
        print(f"[!] Upload failed (HTTP {r.status_code}): {r.text}")
        sys.exit(1)

    mri_id = r.json().get("id")
    print(f" [OK] Upload successful! Assigned MRI ID: {mri_id}")

    # 4. Wait for processing & verify database
    print("\n[4/4] Waiting for AI pipeline to process the scan...")
    print("      (Composition API -> MinIO -> Redpanda -> nnUNet (WSL2 GPU) -> Postgres)")

    success = False
    for attempt in range(1, 30):
        time.sleep(2)
        # Query PostgreSQL node table via Docker CLI
        cmd = [
            "docker", "exec", "postgres", "psql", "-U", "postgres", "-d", "examdb", "-t", "-A", "-c",
            f"SELECT count(*) FROM node WHERE mri_id = '{mri_id}';"
        ]
        try:
            res = subprocess.run(cmd, capture_output=True, text=True, check=True)
            node_count = res.stdout.strip()
        except Exception as e:
            print(f"      [Query Warning]: {e}")
            continue

        if node_count.isdigit() and int(node_count) > 0:
            success = True
            print(f"\n 🎉 SUCCESS! Processing completed in ~{attempt * 2} seconds.")
            print(f"    Found {node_count} detected tumor formation node(s) in Postgres.")
            break
        else:
            print(f"      - Poll {attempt}/30: Still processing... (nodes found: {node_count})")

    if not success:
        print("\n[!] Timeout waiting for processing to complete.")
        print("    Check logs: `docker logs exam_service` and `logs/ml_service.log`.")
        sys.exit(1)

    # Display segment details from Postgres
    print("\n------------------ Postgres Mask & Contour Details ------------------")
    cmd_details = [
        "docker", "exec", "postgres", "psql", "-U", "postgres", "-d", "examdb", "-c",
        f"SELECT n.id as node_id, n.ai, s.id as segment_id, s.image_id, substring(s.contor::text, 1, 45) as contor_preview FROM node n JOIN segment s ON n.id = s.node_id WHERE n.mri_id = '{mri_id}';"
    ]
    res_details = subprocess.run(cmd_details, capture_output=True, text=True)
    print(res_details.stdout)
    print("---------------------------------------------------------------------")
    print(f"View the study in your frontend viewer at: http://localhost:3005")


if __name__ == "__main__":
    main()

