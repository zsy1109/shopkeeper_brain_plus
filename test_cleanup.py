"""List and optionally clean up existing files before E2E test"""
import requests
import json

BASE = "http://localhost:8000"

# List all files
r = requests.get(f"{BASE}/files")
data = r.json()
files = data.get("files", [])

print(f"Total files: {len(files)}")
for f in files:
    fid = f.get("file_id", "?")
    fname = f.get("filename", "?")
    ftitle = f.get("file_title", "?")
    md5 = f.get("md5_hash", "?")
    print(f"  file_id={fid}  filename={fname}  file_title={ftitle}")

# Delete all '万用表' files
for f in files:
    fname = f.get("filename", "")
    fid = f.get("file_id", "")
    if "万用表" in fname:
        print(f"\nDeleting file_id={fid} ({fname})...")
        r = requests.delete(f"{BASE}/files/{fid}")
        result = r.json()
        print(f"  Result: {json.dumps(result, ensure_ascii=False, indent=2)}")

# Verify empty
r2 = requests.get(f"{BASE}/files")
data2 = r2.json()
files2 = data2.get("files", [])
wan_files = [f for f in files2 if "万用表" in f.get("filename", "")]
print(f"\nAfter cleanup - remaining '万用表' files: {len(wan_files)}")