"""文件上传异常校验 — 单元自测脚本

测试用例:
    1. 空 .txt 文件 → 预期: 400, 后缀拦截
    2. 损坏 PDF      → 预期: 400, 损坏拦截
    3. 加密 PDF      → 预期: 400, 加密拦截
    4. 正常 PDF      → 预期: 200, 正常入库
"""
import json
import os
import sys
import tempfile
import time
import urllib.error
import urllib.request

from datetime import datetime

BASE_URL = "http://127.0.0.1:8000"

# ── 测试文件准备 ──
TEST_DIR = os.path.dirname(os.path.abspath(__file__))
NORMAL_PDF = os.path.join(TEST_DIR, "..", "..", "..", "docs", "万用表的使用_origin.pdf")

DAMAGED_PDF = os.path.join(tempfile.gettempdir(), "test_damaged.pdf")
ENCRYPTED_PDF = os.path.join(tempfile.gettempdir(), "test_encrypted.pdf")
EMPTY_TXT = os.path.join(tempfile.gettempdir(), "test_empty.txt")


def prepare_test_files():
    """生成损坏 PDF、加密 PDF、空 txt。"""
    from pypdf import PdfWriter, PdfReader

    # ── 损坏 PDF（写入半截 PDF 头）──
    with open(DAMAGED_PDF, "wb") as f:
        f.write(b"%PDF-1.4\n%%EOF")  # 只有文件头/尾，无内容 → 损坏

    # ── 加密 PDF ──
    writer = PdfWriter()
    writer.add_blank_page(width=595, height=842)  # A4
    writer.encrypt(user_password="123456", owner_password="ownerpass")
    with open(ENCRYPTED_PDF, "wb") as f:
        writer.write(f)

    # ── 空 txt ──
    with open(EMPTY_TXT, "w", encoding="utf-8") as f:
        f.write("")  # 0 字节文本文件


def _upload(filepath: str, filename_override: str = None):
    """用 multipart/form-data 上传单个文件，返回 (status, response_dict)。"""
    boundary = "----TestBoundary" + str(int(time.time() * 1000))
    filename = filename_override or os.path.basename(filepath)
    with open(filepath, "rb") as fh:
        body = fh.read()

    body_bytes = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="file"; filename="{filename}"\r\n'
        f"Content-Type: application/octet-stream\r\n\r\n"
    ).encode("utf-8") + body + f"\r\n--{boundary}--\r\n".encode("utf-8")

    req = urllib.request.Request(
        f"{BASE_URL}/upload",
        data=body_bytes,
        method="POST",
        headers={
            "Content-Type": f"multipart/form-data; boundary={boundary}",
            "Accept": "application/json; charset=utf-8",
        },
    )
    try:
        resp = urllib.request.urlopen(req, timeout=60)
        return resp.status, json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        try:
            detail = json.loads(body).get("detail", body)
        except json.JSONDecodeError:
            detail = body
        return e.code, {"detail": detail}


def print_separator(title: str):
    print(f"\n{'─' * 60}\n  {title}\n{'─' * 60}")


def main():
    print_separator("文件上传异常校验 — 单元自测")
    print(f"  时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  接口: {BASE_URL}/upload")

    prepare_test_files()
    print(f"\n  测试文件已准备:")
    print(f"    损坏PDF:  {DAMAGED_PDF}")
    print(f"    加密PDF:  {ENCRYPTED_PDF}")
    print(f"    空TXT:    {EMPTY_TXT}")
    print(f"    正常PDF:  {NORMAL_PDF}")

    results = []

    # ── 测试用例 1：空 .txt ──
    print_separator("测试用例1: 空 .txt 文件 → 预期 400 (后缀拦截)")
    code, resp = _upload(EMPTY_TXT)
    detail = resp.get("detail", "")
    ok = code == 400 and (".pdf" in detail.lower() or "格式" in detail or "后缀" in detail)
    print(f"  HTTP {code}: {detail}")
    print(f"  结果: {'✅ 通过' if ok else '❌ 失败'}")
    results.append(("空 .txt 后缀拦截", ok, f"HTTP {code}: {detail}"))

    # ── 测试用例 2：损坏 PDF ──
    print_separator("测试用例2: 损坏 PDF → 预期 400 (损坏拦截)")
    code, resp = _upload(DAMAGED_PDF, filename_override="damaged.pdf")
    detail = resp.get("detail", "")
    ok = code == 400 and ("损坏" in detail or "无法正常" in detail or "无效" in detail)
    print(f"  HTTP {code}: {detail}")
    print(f"  结果: {'✅ 通过' if ok else '❌ 失败'}")
    results.append(("损坏 PDF 拦截", ok, f"HTTP {code}: {detail}"))

    # ── 测试用例 3：加密 PDF ──
    print_separator("测试用例3: 加密 PDF → 预期 400 (加密拦截)")
    code, resp = _upload(ENCRYPTED_PDF, filename_override="encrypted.pdf")
    detail = resp.get("detail", "")
    ok = code == 400 and ("加密" in detail)
    print(f"  HTTP {code}: {detail}")
    print(f"  结果: {'✅ 通过' if ok else '❌ 失败'}")
    results.append(("加密 PDF 拦截", ok, f"HTTP {code}: {detail}"))

    # ── 测试用例 4：正常 PDF ──
    print_separator("测试用例4: 正常 PDF → 预期 200 (正常入库)")
    code, resp = _upload(NORMAL_PDF)
    ok = code == 200
    task_id = resp.get("task_id", "") if ok else ""
    print(f"  HTTP {code}, task_id={task_id}")
    print(f"  结果: {'✅ 通过' if ok else '❌ 失败'}")
    results.append(("正常 PDF 入库", ok, f"HTTP {code}, task_id={task_id}"))

    # ── 汇总 ──
    print_separator("测试汇总")
    passed = 0
    failed = 0
    for name, ok, msg in results:
        status = "✅" if ok else "❌"
        if ok:
            passed += 1
        else:
            failed += 1
        print(f"  {status} {name}: {msg}")
    print(f"\n  通过: {passed}/{len(results)}, 失败: {failed}/{len(results)}")

    # ── 清理临时文件 ──
    for p in [DAMAGED_PDF, ENCRYPTED_PDF, EMPTY_TXT]:
        try:
            os.remove(p)
        except OSError:
            pass

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())