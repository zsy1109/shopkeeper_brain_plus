"""End-to-end RAG test for shopkeeper_brain_plus"""
import requests
import time
import json
import sys

BASE_IMPORT = "http://localhost:8000"
BASE_QUERY = "http://localhost:8001"
PDF_PATH = r"D:\pycharm_projects\shopkeeper_brain_plus\docs\万用表的使用_origin.pdf"

results = []

def report(num, name, status, detail=""):
    results.append((num, name, status, detail))
    print(f"\n{'='*60}")
    print(f"Test {num}: {name}")
    print(f"Result: {status}")
    if detail:
        print(f"Detail: {detail}")
    print(f"{'='*60}")

def wait_upload(task_id, timeout=180):
    """Poll /status/{task_id} until completed or timeout"""
    start = time.time()
    while time.time() - start < timeout:
        try:
            r = requests.get(f"{BASE_IMPORT}/status/{task_id}", timeout=5)
            data = r.json()
            status = data.get("status", "")
            if status == "completed":
                return data
            elif status == "failed":
                return data
            time.sleep(2)
        except Exception as e:
            print(f"  [poll error] {e}")
            time.sleep(2)
    return {"status": "timeout"}


# ---- Test 1: Upload PDF ----
print("\n" + "#"*60)
print("# Test 1: Upload PDF")
print("#"*60)

with open(PDF_PATH, "rb") as f:
    r = requests.post(f"{BASE_IMPORT}/upload", files={"file": ("万用表的使用_origin.pdf", f, "application/pdf")})

print(f"HTTP {r.status_code}")
print(f"Response: {json.dumps(r.json(), ensure_ascii=False, indent=2)}")

if r.status_code == 200:
    upload_resp = r.json()
    task_id_1 = upload_resp["task_id"]
    print(f"\nTask ID: {task_id_1}")
    print("Waiting for import to complete...")
    status_data = wait_upload(task_id_1)
    print(f"Final status: {json.dumps(status_data, ensure_ascii=False, indent=2)}")

    if status_data.get("status") == "completed":
        report(1, "Upload PDF 入库", "PASS",
               f"Task {task_id_1} completed, nodes={status_data.get('done_list','')}, "
               f"durations={status_data.get('durations','')}")
    else:
        report(1, "Upload PDF 入库", "FAIL",
               f"Status={status_data.get('status')}, msg={status_data.get('message','')}")
else:
    report(1, "Upload PDF 入库", "FAIL", f"HTTP {r.status_code}: {r.text}")
    sys.exit(1)

# After upload, get the file listing to obtain file_id
r_files = requests.get(f"{BASE_IMPORT}/files")
files_data = r_files.json()
print(f"\nCurrent files in knowledge base: {json.dumps(files_data, ensure_ascii=False, indent=2)}")

# Find our file
file_id = None
file_title = None
for f_item in files_data.get("files", []):
    if "万用表" in f_item.get("filename", ""):
        file_id = f_item.get("file_id")
        file_title = f_item.get("file_title", "")
        break

if not file_id:
    report(1, "Upload PDF 入库 (find file)", "FAIL", "Cannot find uploaded file in /files list")
    sys.exit(1)
print(f"Found file_id={file_id}, file_title={file_title}")

# ---- Test 2: Query the document ----
print("\n" + "#"*60)
print("# Test 2: Query RAG")
print("#"*60)

query_payload = {"query": "万用表有什么功能？怎么使用？", "is_stream": False}
r = requests.post(f"{BASE_QUERY}/query", json=query_payload)
print(f"HTTP {r.status_code}")
query_resp = r.json()
print(f"Response: {json.dumps(query_resp, ensure_ascii=False, indent=2)}")

if r.status_code == 200:
    answer = query_resp.get("answer", "")
    session_id = query_resp.get("session_id", "")
    if answer and len(answer) > 20:
        report(2, "Query RAG 召回", "PASS",
               f"Answer length={len(answer)}, session={session_id}, preview={answer[:200]}...")
    else:
        report(2, "Query RAG 召回", "FAIL",
               f"Answer too short or empty: '{answer}'")
else:
    report(2, "Query RAG 召回", "FAIL", f"HTTP {r.status_code}: {r.text}")

# ---- Test 3: Re-upload same PDF (duplicate check) ----
print("\n" + "#"*60)
print("# Test 3: Re-upload SAME PDF (duplicate bug check)")
print("#"*60)

with open(PDF_PATH, "rb") as f:
    r = requests.post(f"{BASE_IMPORT}/upload", files={"file": ("万用表的使用_origin.pdf", f, "application/pdf")})

print(f"HTTP {r.status_code}")
dup_resp = r.json() if r.text else {}
print(f"Response: {json.dumps(dup_resp, ensure_ascii=False, indent=2)}")

# Check if dedup works (should return 409 if MD5 duplicate detected)
if r.status_code == 409:
    report(3, "重复上传去重", "PASS", f"HTTP 409, dedup working: {dup_resp.get('detail','')}")
elif r.status_code == 200:
    # Potentially created a duplicate - monitor how many chunks & item names exist
    task_id_3 = dup_resp.get("task_id")
    if task_id_3:
        print("Waiting for duplicate import to complete...")
        status_data = wait_upload(task_id_3)
        print(f"Final status: {json.dumps(status_data, ensure_ascii=False, indent=2)}")

    # Check chunks count
    try:
        r_chunks = requests.get(f"{BASE_IMPORT}/files/{file_id}/chunks")
        chunks_info = r_chunks.json()
        chunk_count = chunks_info.get("chunk_count", -1)
        print(f"Chunks after re-upload: {chunk_count}")
    except:
        chunk_count = -1

    # Check files list - how many records for same filename?
    r_files2 = requests.get(f"{BASE_IMPORT}/files")
    files2 = r_files2.json()
    matching = [f for f in files2.get("files", []) if "万用表" in f.get("filename", "")]
    print(f"Matching file records: {len(matching)}")
    for m in matching:
        print(f"  file_id={m.get('file_id')}, hash={m.get('md5_hash','?')}")

    report(3, "重复上传去重", "FAIL ⚠️",
           f"HTTP 200, possible duplicate created. {len(matching)} records, {chunk_count} chunks. "
           f"Need manual verification of Milvus/MongoDB for dirty data.")
else:
    report(3, "重复上传去重", "FAIL", f"Unexpected HTTP {r.status_code}: {r.text}")

# ---- Test 4: Delete document ----
print("\n" + "#"*60)
print("# Test 4: Delete document (3-layer cleanup)")
print("#"*60)

r = requests.delete(f"{BASE_IMPORT}/files/{file_id}")
print(f"HTTP {r.status_code}")
del_resp = r.json()
print(f"Response: {json.dumps(del_resp, ensure_ascii=False, indent=2)}")

if r.status_code == 200 and del_resp.get("success"):
    details = del_resp.get("details", {})
    report(4, "删除文档三层联动", "PASS",
           f"Mongo={details.get('mongo_record')}, Milvus_chunks={details.get('milvus_chunks')}, "
           f"Milvus_item_names={details.get('milvus_item_names')}, MinIO={details.get('minio_object')}")
else:
    report(4, "删除文档三层联动", "FAIL", f"Success=False, message={del_resp.get('message','?')}")

# ---- Test 5: Query after delete ----
print("\n" + "#"*60)
print("# Test 5: Query after delete (should NOT find deleted doc)")
print("#"*60)

query2 = {"query": "万用表有什么功能？", "is_stream": False}
r = requests.post(f"{BASE_QUERY}/query", json=query2)
print(f"HTTP {r.status_code}")
q2_resp = r.json()
print(f"Response: {json.dumps(q2_resp, ensure_ascii=False, indent=2)}")

answer2 = q2_resp.get("answer", "")
# After deletion, answer should be shorter (no RAG context) or indicate not found
if "万用表" not in answer2 or "无法" in answer2 or len(answer2) < 30:
    report(5, "删除后不可检索", "PASS", f"Answer no longer contains document content. Answer: '{answer2[:200]}'")
else:
    report(5, "删除后不可检索", "FAIL ⚠️",
           f"Answer still references deleted document content! Answer: '{answer2[:300]}'")


# ---- Final Summary ----
print("\n\n" + "#"*60)
print("# FINAL SUMMARY")
print("#"*60)
for num, name, status, detail in results:
    print(f"  [{status}] Test {num}: {name}")
    if detail:
        print(f"         {detail}")