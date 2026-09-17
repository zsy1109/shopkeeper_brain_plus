import hashlib
import logging
import os
from datetime import datetime
from typing import List, Dict, Any, Optional

from knowledge.utils.client.storage_clients import StorageClients

logger = logging.getLogger(__name__)

_COLLECTION_NAME = "import_record"


def _get_collection():
    return StorageClients.get_mongo_db()[_COLLECTION_NAME]


def find_duplicate_by_md5(filename: str, md5_hash: str) -> Optional[Dict[str, Any]]:
    try:
        doc = _get_collection().find_one({
            "filename": filename,
            "file_md5": md5_hash,
        })
        return doc
    except Exception as e:
        logger.error(f"[import_record] 查重查询失败: {e}")
        return None


def create_import_record(
        task_id: str,
        filename: str,
        file_title: str,
        item_name: str = "",
        chunk_count: int = 0,
        status: str = "completed",
        minio_object_path: str = "",
        import_file_path: str = "",
        file_md5: str = "",
) -> str:
    doc = {
        "task_id": task_id,
        "filename": filename,
        "file_title": file_title,
        "item_name": item_name,
        "chunk_count": chunk_count,
        "status": status,
        "minio_object_path": minio_object_path,
        "import_file_path": import_file_path,
        "file_md5": file_md5,
        "import_time": datetime.now().timestamp(),
    }
    try:
        result = _get_collection().insert_one(doc)
        logger.info(f"[import_record] 已入库: file={filename}, chunks={chunk_count}, _id={result.inserted_id}")
        return str(result.inserted_id)
    except Exception as e:
        logger.error(f"[import_record] 写入失败: {e}")
        return ""


def update_import_record(file_id: str, **fields) -> bool:
    from bson import ObjectId
    try:
        result = _get_collection().update_one(
            {"_id": ObjectId(file_id)},
            {"$set": fields},
        )
        return result.modified_count > 0
    except Exception as e:
        logger.error(f"[import_record] 更新失败: {e}")
        return False


def list_import_records(limit: int = 200) -> List[Dict[str, Any]]:
    from pymongo import DESCENDING
    try:
        cursor = (
            _get_collection()
            .find({})
            .sort("import_time", DESCENDING)
            .limit(limit)
        )
        results = []
        for doc in cursor:
            doc["file_id"] = str(doc["_id"])
            del doc["_id"]
            results.append(doc)
        return results
    except Exception as e:
        logger.error(f"[import_record] 列表查询失败: {e}")
        return []


def get_import_record(file_id: str) -> Optional[Dict[str, Any]]:
    from bson import ObjectId
    try:
        doc = _get_collection().find_one({"_id": ObjectId(file_id)})
        if not doc:
            return None
        doc["file_id"] = str(doc["_id"])
        del doc["_id"]
        return doc
    except Exception as e:
        logger.error(f"[import_record] 详情查询失败: {e}")
        return None


def delete_import_record(file_id: str) -> bool:
    from bson import ObjectId
    try:
        result = _get_collection().delete_one({"_id": ObjectId(file_id)})
        ok = result.deleted_count > 0
        logger.info(f"[import_record] 删除 file_id={file_id}, ok={ok}")
        return ok
    except Exception as e:
        logger.error(f"[import_record] 删除失败: {e}")
        return False


def get_bucket_name() -> str:
    return os.getenv("MINIO_BUCKET_NAME", "")