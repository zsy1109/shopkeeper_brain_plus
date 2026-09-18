"""
MD5 文件查重功能验证脚本

用法：
    cd knowledge
    python -m test.import.test_dedup

测试场景：
    1. 计算同一文件两次，MD5 应当一致
    2. 计算不同文件，MD5 应当不同
    3. 模拟查重逻辑：相同 MD5 应被拦截
    4. 空文件 / 损坏文件：MD5 仍可计算，但会被原有文件大小校验拦截
"""

import hashlib
import io
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def calculate_md5_from_bytes(data: bytes) -> str:
    md5 = hashlib.md5()
    md5.update(data)
    return md5.hexdigest()


def calculate_md5_from_file(file_path: str) -> str:
    md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            md5.update(chunk)
    return md5.hexdigest()


class FakeCollection:
    """模拟 MongoDB collection 用于单元测试查重逻辑"""

    def __init__(self):
        self._records = {}

    def insert_one(self, doc):
        from bson import ObjectId
        doc["_id"] = ObjectId()
        md5 = doc.get("file_md5", "")
        self._records[md5] = doc
        return type("InsertResult", (), {"inserted_id": doc["_id"]})()

    def find_one(self, query):
        md5 = query.get("file_md5", "")
        return self._records.get(md5)


def test_md5_consistency():
    """测试1：同一内容两次计算 MD5 应当一致"""
    print("\n" + "=" * 60)
    print("测试1: MD5 一致性")
    content = b"Hello PDF World! This is test content." * 100
    hash1 = calculate_md5_from_bytes(content)
    hash2 = calculate_md5_from_bytes(content)
    assert hash1 == hash2, f"MD5 不一致: {hash1} != {hash2}"
    print(f"  ✓ 同一内容 MD5 一致: {hash1}")
    return True


def test_md5_different_content():
    """测试2：不同内容 MD5 应当不同"""
    print("\n" + "=" * 60)
    print("测试2: 不同内容 MD5 区分")
    content_a = b"PDF content A" * 100
    content_b = b"PDF content B" * 100
    hash_a = calculate_md5_from_bytes(content_a)
    hash_b = calculate_md5_from_bytes(content_b)
    assert hash_a != hash_b, "不同内容 MD5 应该不同"
    print(f"  ✓ 内容A MD5: {hash_a}")
    print(f"  ✓ 内容B MD5: {hash_b}")
    print(f"  ✓ 不同内容的 MD5 确实不同")
    return True


def test_duplicate_interception():
    """测试3：模拟查重逻辑 - 相同 MD5 被拦截"""
    print("\n" + "=" * 60)
    print("测试3: 查重拦截逻辑")
    fake_collection = FakeCollection()

    # 模拟首次上传：写入一条记录
    md5_1 = calculate_md5_from_bytes(b"Real PDF content" * 50)
    fake_collection.insert_one({
        "filename": "test.pdf",
        "file_md5": md5_1,
        "status": "completed",
    })
    print(f"  ✓ 首次上传: {md5_1} 已入库")

    # 模拟第二次上传相同文件：查重应当找到
    duplicate = fake_collection.find_one({"file_md5": md5_1})
    assert duplicate is not None, "重复文件应被检测到"
    print(f"  ✓ 第二次上传相同文件: 已被拦截 (查到记录 _id={duplicate['_id']})")

    # 模拟上传不同文件：查重不应找到
    md5_2 = calculate_md5_from_bytes(b"Another PDF content" * 50)
    duplicate_2 = fake_collection.find_one({"file_md5": md5_2})
    assert duplicate_2 is None, "全新文件不应被拦截"
    print(f"  ✓ 上传全新文件 MD5={md5_2}: 未被拦截，可正常入库")

    return True


def test_empty_file():
    """测试4：空文件仍可计算 MD5（原有文件大小校验在前）"""
    print("\n" + "=" * 60)
    print("测试4: 空文件 MD5 计算")
    empty_md5 = calculate_md5_from_bytes(b"")
    expected = hashlib.md5(b"").hexdigest()
    assert empty_md5 == expected, "空文件 MD5 计算错误"
    print(f"  ✓ 空文件 MD5: {empty_md5} (与标准一致)")
    print(f"  ✓ 注意: 空文件会被文件大小校验(≥1KB)先拦截，不会进入查重")
    return True


def test_file_based_md5():
    """测试5：基于实际文件的 MD5 计算"""
    print("\n" + "=" * 60)
    print("测试5: 基于文件的 MD5 计算")

    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(b"%PDF-1.4 test content for dedup verification" * 100)
        tmp_path = tmp.name

    try:
        md5_file = calculate_md5_from_file(tmp_path)
        md5_bytes = calculate_md5_from_bytes(
            b"%PDF-1.4 test content for dedup verification" * 100
        )
        assert md5_file == md5_bytes, "文件 MD5 与字节 MD5 不一致"
        print(f"  ✓ 文件 MD5: {md5_file}")
        print(f"  ✓ 字节 MD5: {md5_bytes}")
        print(f"  ✓ 两者一致")
    finally:
        os.unlink(tmp_path)

    return True


def test_different_filenames_same_content():
    """测试6：不同文件名相同内容 → 应当被拦截（纯 MD5 查重）"""
    print("\n" + "=" * 60)
    print("测试6: 不同文件名 + 相同内容 → 纯 MD5 查重")
    content = b"Same content different name" * 100
    md5_hash = calculate_md5_from_bytes(content)

    fake_collection = FakeCollection()
    fake_collection.insert_one({
        "filename": "report_v1.pdf",
        "file_md5": md5_hash,
    })
    print(f"  ✓ 已入库: report_v1.pdf (MD5={md5_hash})")

    # 用不同文件名上传相同内容
    duplicate = fake_collection.find_one({"file_md5": md5_hash})
    assert duplicate is not None, "即使文件名不同，相同 MD5 也应当被拦截"
    print(f"  ✓ 上传 report_v2.pdf (内容相同): 被拦截 → 纯 MD5 查重生效")
    return True


def main():
    print("=" * 60)
    print("MD5 文件查重功能 - 验证测试")
    print("=" * 60)

    tests = [
        ("MD5 一致性", test_md5_consistency),
        ("不同内容 MD5 区分", test_md5_different_content),
        ("查重拦截逻辑", test_duplicate_interception),
        ("空文件 MD5 计算", test_empty_file),
        ("基于文件的 MD5 计算", test_file_based_md5),
        ("不同文件名相同内容拦截", test_different_filenames_same_content),
    ]

    passed = 0
    failed = 0
    for name, test_fn in tests:
        try:
            test_fn()
            passed += 1
        except AssertionError as e:
            print(f"  ✗ 失败: {e}")
            failed += 1
        except Exception as e:
            print(f"  ✗ 异常: {type(e).__name__}: {e}")
            failed += 1

    print("\n" + "=" * 60)
    print(f"测试结果: 通过={passed}, 失败={failed}, 总计={passed + failed}")
    print("=" * 60)

    if failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()