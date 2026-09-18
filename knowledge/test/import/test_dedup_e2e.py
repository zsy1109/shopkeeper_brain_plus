"""
MD5 查重功能 - 端到端全量测试

前置条件：
    1. FastAPI 服务器已启动（python -m knowledge.api.import_router）
    2. MongoDB / Milvus / MinIO 服务正常运行

用法：
    cd D:\pycharm_projects\shopkeeper_brain_plus
    python -m knowledge.test.import.test_dedup_e2e

测试步骤：
    1. 记录 MongoDB / Milvus 基准数据
    2. 上传 PDF A → 等待入库完成 → 校验记录新增
    3. 再次上传同一 PDF A → 校验 HTTP 409 拦截 → 校验无新增记录
    4. 上传 PDF B（不同文件）→ 等待入库完成 → 校验记录新增
    5. 输出测试报告
"""

import hashlib
import json
import os
import sys
import time
import urllib.request
import urllib.error
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

BASE_URL = "http://127.0.0.1:8000"
POLL_INTERVAL = 3  # 轮询间隔（秒）
MAX_WAIT = 600  # 最大等待时间（秒）

PDF_A_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "processor", "import_processor", "temp_dir", "万用表的使用.pdf",
)

PDF_B_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "processor", "import_processor", "temp_dir", "万用表的使用", "txt", "万用表的使用_origin.pdf",
)


def _http(method: str, path: str, body=None, files=None):
    url = f"{BASE_URL}{path}"
    if body is not None:
        body = json.dumps(body).encode("utf-8")

    req = urllib.request.Request(url, data=body, method=method)
    req.add_header("Accept", "application/json; charset=utf-8")

    if body is not None:
        req.add_header("Content-Type", "application/json; charset=utf-8")

    try:
        resp = urllib.request.urlopen(req, timeout=30)
        return resp.status, json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        body_bytes = e.read()
        try:
            detail = json.loads(body_bytes.decode("utf-8"))
        except Exception:
            detail = body_bytes.decode("utf-8", errors="replace")
        return e.code, detail


def _upload_pdf(file_path: str):
    boundary = "----TestBoundaryDedup"
    filename = os.path.basename(file_path)
    with open(file_path, "rb") as f:
        file_data = f.read()

    body = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="file"; filename="{filename}"\r\n'
        f"Content-Type: application/pdf\r\n\r\n"
    ).encode("utf-8") + file_data + f"\r\n--{boundary}--\r\n".encode("utf-8")

    req = urllib.request.Request(f"{BASE_URL}/upload", data=body, method="POST")
    req.add_header("Accept", "application/json; charset=utf-8")
    req.add_header("Content-Type", f"multipart/form-data; boundary={boundary}")

    try:
        resp = urllib.request.urlopen(req, timeout=30)
        return resp.status, json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        body_bytes = e.read()
        try:
            detail = json.loads(body_bytes.decode("utf-8"))
        except Exception:
            detail = {"detail": body_bytes.decode("utf-8", errors="replace")}
        return e.code, detail


def _poll_status(task_id: str, timeout: int = MAX_WAIT):
    start = time.time()
    while time.time() - start < timeout:
        code, data = _http("GET", f"/status/{task_id}")
        if code == 200 and data.get("status") in ("completed", "failed"):
            return data
        time.sleep(POLL_INTERVAL)
    return {"status": "timeout"}


def _list_files():
    code, data = _http("GET", "/files")
    return data if code == 200 else {"total": 0, "items": []}


def _md5_file(file_path: str) -> str:
    md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            md5.update(chunk)
    return md5.hexdigest()


def _check_server():
    try:
        code, _ = _http("GET", "/")
        return code == 200
    except Exception:
        return False


def main():
    print("=" * 65)
    print("  MD5 查重 - 端到端全量测试")
    print(f"  时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 65)

    # ── 前置检查 ──
    print("\n[前置] 检查服务器连通性...")
    if not _check_server():
        print("  ✗ 服务器未启动！请先运行:")
        print("    cd D:\\pycharm_projects\\shopkeeper_brain_plus")
        print("    python -m knowledge.api.import_router")
        return 1
    print("  ✓ 服务器已就绪")

    print("\n[前置] 检查 PDF A...")
    if not os.path.exists(PDF_A_PATH):
        print(f"  ✗ PDF A 不存在: {PDF_A_PATH}")
        return 1
    md5_a = _md5_file(PDF_A_PATH)
    print(f"  ✓ PDF A: {os.path.basename(PDF_A_PATH)}  (MD5={md5_a})")
    print(f"    大小: {os.path.getsize(PDF_A_PATH)} bytes")

    print("\n[前置] 检查 PDF B...")
    if not os.path.exists(PDF_B_PATH):
        print(f"  ✗ PDF B 不存在: {PDF_B_PATH}")
        return 1
    md5_b = _md5_file(PDF_B_PATH)
    print(f"  ✓ PDF B: {os.path.basename(PDF_B_PATH)}  (MD5={md5_b})")
    print(f"    大小: {os.path.getsize(PDF_B_PATH)} bytes")

    # ── 基准数据 ──
    print("\n[基准] 记录当前 MongoDB 文件数...")
    before = _list_files()
    before_count = before.get("total", 0)
    print(f"  ✓ 现有文件记录数: {before_count}")

    # ── 步骤 1 ──
    print("\n" + "-" * 65)
    print("  步骤1: 上传 PDF A（首次，应正常入库）")
    print("-" * 65)

    code1, resp1 = _upload_pdf(PDF_A_PATH)
    print(f"  HTTP 状态码: {code1}")
    print(f"  响应: {json.dumps(resp1, ensure_ascii=False)}")

    step1_pass = code1 == 200
    task_id_a = resp1.get("task_id", "")

    if step1_pass:
        print(f"  ✓ 上传成功，task_id={task_id_a}")
        print(f"  等待入库完成 (最长 {MAX_WAIT}s)...")
        status1 = _poll_status(task_id_a)
        print(f"  最终状态: {status1.get('status')}")
        if status1.get("status") != "completed":
            step1_pass = False
            print(f"  ✗ 入库失败或超时")
    else:
        print(f"  ✗ 上传失败: {resp1}")

    # ── 步骤 2 ──
    print("\n" + "-" * 65)
    print("  步骤2: 再次上传同一 PDF A（应被 MD5 查重拦截）")
    print("-" * 65)

    code2, resp2 = _upload_pdf(PDF_A_PATH)
    print(f"  HTTP 状态码: {code2}")
    print(f"  响应: {json.dumps(resp2, ensure_ascii=False)}")

    step2_pass = code2 == 409
    if step2_pass:
        detail_msg = resp2.get("detail", "") if isinstance(resp2, dict) else str(resp2)
        if "已上传至知识库" in detail_msg or "请勿重复上传" in detail_msg:
            print(f"  ✓ 正确拦截，返回409，消息包含预期提示")
        else:
            print(f"  △ 返回409但消息不符预期: {detail_msg}")
    else:
        print(f"  ✗ 期望 409，实际 {code2}")

    # ── 校验：MongoDB 无新增记录 ──
    print("\n[校验] 检查 MongoDB 记录数（步骤2后）...")
    mid = _list_files()
    mid_count = mid.get("total", 0)
    expected_mid = before_count + 1  # 只有步骤1新增了1条
    count_ok = mid_count == expected_mid
    print(f"  期望: {expected_mid}  实际: {mid_count}  {'✓' if count_ok else '✗'}")
    step2_pass = step2_pass and count_ok

    # ── 步骤 3 ──
    print("\n" + "-" * 65)
    print("  步骤3: 上传 PDF B（不同文件，应正常入库）")
    print("-" * 65)

    code3, resp3 = _upload_pdf(PDF_B_PATH)
    print(f"  HTTP 状态码: {code3}")
    print(f"  响应: {json.dumps(resp3, ensure_ascii=False)}")

    step3_pass = code3 == 200
    task_id_b = resp3.get("task_id", "")

    if step3_pass:
        print(f"  ✓ 上传成功，task_id={task_id_b}")
        print(f"  等待入库完成 (最长 {MAX_WAIT}s)...")
        status3 = _poll_status(task_id_b)
        print(f"  最终状态: {status3.get('status')}")
        if status3.get("status") != "completed":
            step3_pass = False
            print(f"  ✗ 入库失败或超时")
    else:
        print(f"  ✗ 上传失败: {resp3}")

    # ── 最终校验 ──
    print("\n[校验] 检查最终 MongoDB 记录数...")
    after = _list_files()
    after_count = after.get("total", 0)
    expected_after = before_count + 2  # 步骤1 + 步骤3
    final_count_ok = after_count == expected_after
    print(f"  期望: {expected_after}  实际: {after_count}  {'✓' if final_count_ok else '✗'}")
    step3_pass = step3_pass and final_count_ok

    # ── 测试报告 ──
    results = [
        ("步骤1: PDF A 首次上传入库", step1_pass),
        ("步骤2: PDF A 重复上传拦截", step2_pass),
        ("步骤3: PDF B 正常入库", step3_pass),
    ]

    print("\n" + "=" * 65)
    print("  测 试 报 告")
    print("=" * 65)
    for desc, ok in results:
        print(f"  {'✓ 通过' if ok else '✗ 失败'}  {desc}")

    passed = sum(1 for _, ok in results if ok)
    total = len(results)
    print(f"\n  结果: {passed}/{total} 通过")
    print(f"  PDF A  MD5: {md5_a}")
    print(f"  PDF B  MD5: {md5_b}")
    print("=" * 65)

    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())