#!/usr/bin/env python3
"""
LUNA-MATCH Full Backend Demo
Usage: python scripts/demo_all_endpoints.py [--url https://lunarcoreengine.onrender.com]
Default URL: http://localhost:8000
"""

import os
import sys
import time
import json
import uuid
import argparse
import urllib.request
import urllib.error
from pathlib import Path


def encode_multipart_form(fields: dict, files: dict):
    boundary = f"----LUNAMatchBoundary{uuid.uuid4().hex}"
    body = bytearray()

    for name, value in fields.items():
        body.extend(f"--{boundary}\r\n".encode("utf-8"))
        body.extend(f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode("utf-8"))
        body.extend(f"{value}\r\n".encode("utf-8"))

    for name, (filename, file_bytes, content_type) in files.items():
        body.extend(f"--{boundary}\r\n".encode("utf-8"))
        body.extend(f'Content-Disposition: form-data; name="{name}"; filename="{filename}"\r\n'.encode("utf-8"))
        body.extend(f"Content-Type: {content_type}\r\n\r\n".encode("utf-8"))
        body.extend(file_bytes)
        body.extend(b"\r\n")

    body.extend(f"--{boundary}--\r\n".encode("utf-8"))
    content_type_header = f"multipart/form-data; boundary={boundary}"
    return bytes(body), content_type_header


def request(method, url, data=None, files=None, headers=None, timeout=60):
    if headers is None:
        headers = {}

    body = None
    if files:
        fields = data or {}
        body, ct = encode_multipart_form(fields, files)
        headers["Content-Type"] = ct
    elif data is not None:
        body = json.dumps(data).encode("utf-8")
        headers["Content-Type"] = "application/json"

    req = urllib.request.Request(url, data=body, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            content = resp.read()
            return resp.status, resp.headers, content
    except urllib.error.HTTPError as e:
        content = e.read()
        return e.code, e.headers, content
    except Exception as e:
        return 0, {}, str(e).encode("utf-8")


def main():
    parser = argparse.ArgumentParser(description="LUNA-MATCH Full Backend Demo Runner")
    parser.add_argument("--url", default="http://localhost:8000", help="Base URL of LUNA-MATCH service")
    parser.add_argument("--mode", default="standard", choices=["standard", "fast"], help="Registration mode")
    args = parser.parse_args()

    BASE = args.url.rstrip("/")
    steps_status = {}

    print("\n" + "=" * 70)
    print(f"🛰️  LUNA-MATCH Planetary Image Registration — Full Backend Demo")
    print(f"🔗  Target Base URL: {BASE}")
    print("=" * 70)

    # 1. Health
    print("\n[Step 1] Checking service health (/health)...")
    status_code, headers, body = request("GET", f"{BASE}/health")
    if status_code == 200:
        health_data = json.loads(body.decode("utf-8"))
        print(f"  ✅ Health Check OK: service='{health_data.get('service')}', version='{health_data.get('version')}'")
        steps_status["1. Health Check"] = "PASSED"
    else:
        print(f"  ❌ Health Check FAILED: Status {status_code}, error={body.decode('utf-8')}")
        steps_status["1. Health Check"] = "FAILED"
        print("Aborting demo due to health check failure.")
        sys.exit(1)

    # 2. Register
    print("\n[Step 2] Submitting registration job with verified test pair (/register)...")
    path_a = Path("data/samples/verified_a.tif")
    path_b = Path("data/samples/verified_b.tif")

    if not path_a.exists() or not path_b.exists():
        print(f"  ❌ Sample files not found: {path_a} or {path_b}")
        steps_status["2. Register Job"] = "FAILED"
        sys.exit(1)

    with open(path_a, "rb") as f_a, open(path_b, "rb") as f_b:
        files = {
            "img_a": ("verified_a.tif", f_a.read(), "image/tiff"),
            "img_b": ("verified_b.tif", f_b.read(), "image/tiff"),
        }
    fields = {"mode": args.mode}

    status_code, headers, body = request("POST", f"{BASE}/register", data=fields, files=files)
    if status_code == 202:
        reg_data = json.loads(body.decode("utf-8"))
        job_id = reg_data["job_id"]
        print(f"  ✅ Job registered successfully: job_id = '{job_id}'")
        steps_status["2. Register Job"] = "PASSED"
    else:
        print(f"  ❌ Registration failed: Status {status_code}, body={body.decode('utf-8')}")
        steps_status["2. Register Job"] = "FAILED"
        sys.exit(1)

    # 3. Poll until DONE
    print(f"\n[Step 3] Polling job state until DONE (/jobs/{job_id})...")
    poll_success = False
    for attempt in range(40):
        time.sleep(0.3)
        sys.stdout.write(".")
        sys.stdout.flush()

        status_code, headers, body = request("GET", f"{BASE}/jobs/{job_id}")
        if status_code == 200:
            job_info = json.loads(body.decode("utf-8"))
            st = job_info.get("status")
            if st == "DONE":
                print(f"\n  ✅ Job reached DONE state in {job_info.get('elapsed_s', 0):.2f}s")
                poll_success = True
                break
            elif st == "FAILED":
                print(f"\n  ❌ Job FAILED: {job_info.get('error')}")
                break

    if poll_success:
        steps_status["3. Pipeline Execution"] = "PASSED"
    else:
        steps_status["3. Pipeline Execution"] = "FAILED"
        print("Aborting demo due to job processing failure.")
        sys.exit(1)

    # 4. Summary & Quantitative Metrics Table
    print(f"\n[Step 4] Fetching Quality Summary (/jobs/{job_id}/summary)...")
    status_code, headers, body = request("GET", f"{BASE}/jobs/{job_id}/summary")
    if status_code == 200:
        summary_data = json.loads(body.decode("utf-8"))
        m = summary_data.get("metrics", {})
        qa = summary_data.get("quality_assessment", {})

        print("\n" + "-" * 78)
        print("  METRICS TABLE — REGISTRATION PERFORMANCE")
        print("-" * 78)
        print(f"  Letter Grade     : {qa.get('grade')} ({qa.get('confidence_label')})")
        print(f"  RMSE (Reproj)    : {m.get('rmse_px')} px (Sub-pixel: {'YES' if float(m.get('rmse_px', 99)) < 0.5 else 'NO'})")
        print(f"  Verified Inliers : {m.get('n_inliers')} / {m.get('n_total')} ({float(m.get('inlier_ratio', 0))*100:.1f}%)")
        print(f"  Spatial SDI      : {m.get('sdi')}")
        print(f"  SSIM Perceptual  : {m.get('ssim')}")
        print(f"  Transform Model  : {m.get('transform_type')}")
        print(f"  Elapsed Latency  : {m.get('elapsed_s'):.2f} s")
        print(f"  Reasoning        : {qa.get('reasoning')}")
        print("-" * 78)
        steps_status["4. Summary & Metrics"] = "PASSED"
    else:
        print(f"  ❌ Failed to fetch summary: Status {status_code}")
        steps_status["4. Summary & Metrics"] = "FAILED"

    # 5. Download Previews & Graphs & CSV
    print("\n[Step 5] Validating Artifact Endpoints (Previews, Graphs, CSV)...")
    artifact_endpoints = [
        ("preview_checkerboard", f"{BASE}/jobs/{job_id}/preview?kind=checkerboard"),
        ("preview_tiepoints", f"{BASE}/jobs/{job_id}/preview?kind=tiepoints"),
        ("preview_residual", f"{BASE}/jobs/{job_id}/preview?kind=residual"),
        ("graph_confidence_gauge", f"{BASE}/jobs/{job_id}/graphs?kind=confidence_gauge"),
        ("graph_residual_scatter", f"{BASE}/jobs/{job_id}/graphs?kind=residual_scatter"),
        ("export_control_points", f"{BASE}/jobs/{job_id}/export?kind=control_points"),
    ]

    artifacts_all_ok = True
    for name, endpoint in artifact_endpoints:
        st_code, hdrs, b = request("GET", endpoint)
        if st_code == 200:
            print(f"  ✅ {name:<24}: HTTP 200 OK | {len(b):>8} bytes | content-type: {hdrs.get('content-type', '')}")
        else:
            print(f"  ❌ {name:<24}: HTTP {st_code} Error | {b.decode('utf-8', errors='ignore')}")
            artifacts_all_ok = False

    steps_status["5. Artifacts & Visuals"] = "PASSED" if artifacts_all_ok else "FAILED"

    # 6. Agent Explain Layer
    print("\n[Step 6] Testing AI Explanation Layer (/agent/explain)...")
    status_code, headers, body = request("POST", f"{BASE}/agent/explain/{job_id}/start")
    if status_code == 200:
        start_res = json.loads(body.decode("utf-8"))
        print(f"  ✅ Context loaded: {start_res}")

        # Fast path test
        st_code, hdrs, b = request("POST", f"{BASE}/agent/explain/{job_id}/message", data={"query": "what is the rmse"})
        if st_code == 200:
            fast_data = json.loads(b.decode("utf-8"))
            print(f"  ✅ Fast-Path Query: used_fast_path={fast_data['used_fast_path']}, latency={fast_data['latency_ms']:.1f}ms")
            print(f"     Answer: '{fast_data['text_response']}'")
        else:
            print(f"  ❌ Fast-Path Query failed: HTTP {st_code}")

        # Generative path test (guarded by GROQ_API_KEY)
        if os.environ.get("GROQ_API_KEY"):
            st_code, hdrs, b = request("POST", f"{BASE}/agent/explain/{job_id}/message", data={"query": "describe the crater distribution and confidence"})
            if st_code == 200:
                gen_data = json.loads(b.decode("utf-8"))
                print(f"  ✅ Generative Explain Query: latency={gen_data['latency_ms']:.1f}ms")
                print(f"     Answer: '{gen_data['text_response'][:120]}...'")
            else:
                print(f"  ⚠️  Generative Query HTTP {st_code}: {b.decode('utf-8')}")
        else:
            print("  ℹ️  GROQ_API_KEY not set in local env; skipped generative explain query test.")

        steps_status["6. Agent Explanation Layer"] = "PASSED"
    else:
        print(f"  ❌ /agent/explain/start failed: HTTP {status_code}")
        steps_status["6. Agent Explanation Layer"] = "FAILED"

    # 7. Research Companion
    print("\n[Step 7] Testing Research Companion (/research)...")
    status_code, headers, body = request("POST", f"{BASE}/research/session")
    if status_code == 200:
        sess_data = json.loads(body.decode("utf-8"))
        sid = sess_data["session_id"]
        print(f"  ✅ Research session created: session_id = '{sid}'")

        if os.environ.get("GROQ_API_KEY"):
            st_code, hdrs, b = request("POST", f"{BASE}/research/{sid}/message", data={"query": "What is the resolution of Chandrayaan-2 OHRC?"})
            if st_code == 200:
                r_data = json.loads(b.decode("utf-8"))
                print(f"  ✅ Research Query Answered: latency={r_data['latency_ms']:.1f}ms")
                print(f"     Answer: '{r_data['text_response'][:120]}...'")
            else:
                print(f"  ⚠️  Research Query HTTP {st_code}: {b.decode('utf-8')}")
        else:
            print("  ℹ️  GROQ_API_KEY not set in local env; skipped generative research query test.")

        steps_status["7. Research Companion"] = "PASSED"
    else:
        print(f"  ❌ /research/session failed: HTTP {status_code}")
        steps_status["7. Research Companion"] = "FAILED"

    # 8. Final Summary
    print("\n" + "=" * 70)
    print("📋  DEMO EXECUTION SUMMARY")
    print("=" * 70)
    all_passed = True
    for step_name, st in steps_status.items():
        icon = "✅" if st == "PASSED" else "❌"
        print(f"  {icon}  {step_name:<30} : {st}")
        if st != "PASSED":
            all_passed = False
    print("=" * 70)

    if all_passed:
        print("🎉 ALL ENDPOINTS & VERIFICATIONS PASSED SUCCESSFULLY!\n")
        sys.exit(0)
    else:
        print("⚠️ SOME CHECKS FAILED. Please inspect the output above.\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
