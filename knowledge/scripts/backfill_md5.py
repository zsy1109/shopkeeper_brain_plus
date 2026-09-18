"""
后台脚本：批量计算历史文档的 MD5 并补入 MongoDB 的 file_md5 字段

用法：
    cd knowledge
    python -m scripts.backfill_md5

功能：
    1. 遍历 MongoDB 中所有缺少 file_md5 的历史导入记录
    2. 优先从本地 import_file_path 读取文件计算 MD5
    3. 本地文件不存在时，尝试从 MinIO 下载后再计算 MD5
    4. 更新 MongoDB 中对应记录的 file_md5 字段
"""

import hashlib
import logging
import os
import sys
import tempfile
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from knowledge.utils.client.storage_clients import StorageClients

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


COLLECTION_NAME = "import_record"


def _get_collection():
    return StorageClients.get_mongo_db()[COLLECTION_NAME]


def calculate_md5(file_path: str) -> str:
    md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            md5.update(chunk)
    return md5.hexdigest()


def download_from_minio(minio_object_path: str) -> str | None:
    try:
        minio_client = StorageClients.get_minio_client()
    except Exception as e:
        logger.warning(f"无法连接 MinIO: {e}")
        return None

    bucket_name = os.getenv("MINIO_BUCKET_NAME", "")
    if not bucket_name:
        logger.warning("MINIO_BUCKET_NAME 环境变量未设置")
        return None

    suffix = os.path.splitext(minio_object_path)[1] or ".pdf"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp_path = tmp.name

    try:
        minio_client.fget_object(bucket_name, minio_object_path, tmp_path)
        logger.info(f"已从 MinIO 下载: {minio_object_path} -> {tmp_path}")
        return tmp_path
    except Exception as e:
        logger.warning(f"从 MinIO 下载失败 ({minio_object_path}): {e}")
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        return None


def backfill():
    collection = _get_collection()
    query = {
        "$or": [
            {"file_md5": {"$exists": False}},
            {"file_md5": None},
            {"file_md5": ""},
        ]
    }

    cursor = collection.find(query)
    records = list(cursor)
    total = len(records)

    logger.info(f"共发现 {total} 条缺少 file_md5 的历史记录")

    if total == 0:
        logger.info("没有需要回填的记录，退出")
        return

    success_count = 0
    skip_count = 0
    fail_count = 0
    temp_files = []

    for idx, doc in enumerate(records, start=1):
        doc_id = doc["_id"]
        filename = doc.get("filename", "未知文件")
        import_file_path = doc.get("import_file_path", "")
        minio_object_path = doc.get("minio_object_path", "")

        logger.info(f"[{idx}/{total}] 处理: {filename} (_id={doc_id})")

        md5_hash = None
        source = None

        # 尝试本地文件
        if import_file_path and os.path.exists(import_file_path):
            try:
                md5_hash = calculate_md5(import_file_path)
                source = "local"
                logger.info(f"  ✓ 本地文件计算成功, MD5={md5_hash}")
            except Exception as e:
                logger.warning(f"  ✗ 本地文件计算失败: {e}")

        # 本地不可用则尝试 MinIO
        if md5_hash is None and minio_object_path:
            tmp_path = download_from_minio(minio_object_path)
            if tmp_path:
                temp_files.append(tmp_path)
                try:
                    md5_hash = calculate_md5(tmp_path)
                    source = "minio"
                    logger.info(f"  ✓ MinIO 文件计算成功, MD5={md5_hash}")
                except Exception as e:
                    logger.warning(f"  ✗ MinIO 文件计算失败: {e}")

        # 更新 MongoDB
        if md5_hash:
            try:
                collection.update_one(
                    {"_id": doc_id},
                    {"$set": {"file_md5": md5_hash}},
                )
                success_count += 1
                logger.info(f"  ✓ MongoDB 已更新 (来源: {source})")
            except Exception as e:
                logger.error(f"  ✗ MongoDB 更新失败: {e}")
                fail_count += 1
        else:
            logger.warning(f"  ⊘ 跳过: 无可用文件来源")
            skip_count += 1

    # 清理临时文件
    for tmp_path in temp_files:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass

    logger.info("=" * 50)
    logger.info(f"回填完成: 成功={success_count}, 跳过={skip_count}, 失败={fail_count}, 总计={total}")


if __name__ == "__main__":
    backfill()