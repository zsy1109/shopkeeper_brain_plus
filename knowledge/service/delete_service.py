import logging
import os

from knowledge.utils.client.storage_clients import StorageClients
from knowledge.utils import milvus_util
from knowledge.utils import mongo_import_util

logger = logging.getLogger(__name__)


def delete_document(file_id: str) -> dict:
    """删除已导入文档：MongoDB + Milvus(kb_chunks_v1 + kb_item_names_v1) + MinIO 三层清理"""
    record = mongo_import_util.get_import_record(file_id)
    if not record:
        return {"success": False, "message": f"文件记录不存在 (file_id={file_id})"}

    file_title = record.get("file_title", "")
    item_name = record.get("item_name", "")
    minio_path = record.get("minio_object_path", "")
    filename = record.get("filename", "")
    task_id = record.get("task_id", "")

    logger.info(
        f"[delete] 开始删除: file_id={file_id}, filename={filename}, "
        f"file_title={file_title}, item_name={item_name}, task_id={task_id}"
    )

    chunks_deleted = _delete_milvus(file_title)
    item_names_deleted = _delete_item_names(file_title)
    mongo_ok = mongo_import_util.delete_import_record(file_id)
    minio_ok = _delete_minio(minio_path)

    milvus_ok = chunks_deleted >= 0 and item_names_deleted >= 0

    logger.info(
        f"[delete] 删除结果: chunks={chunks_deleted}, item_names={item_names_deleted}, "
        f"mongo={mongo_ok}, minio={minio_ok}"
    )

    all_ok = milvus_ok and mongo_ok and minio_ok

    if all_ok:
        message = "删除成功"
    elif mongo_ok and not milvus_ok:
        message = "本地记录已删除，但向量库删除失败，请重试"
    elif mongo_ok and not minio_ok:
        message = "本地记录已删除，但对象存储删除失败，请检查"
    else:
        message = "部分组件删除失败，请检查日志"

    return {
        "success": all_ok,
        "message": message,
        "filename": filename,
        "file_title": file_title,
        "item_name": item_name,
        "details": {
            "milvus_chunks": chunks_deleted,
            "milvus_item_names": item_names_deleted,
            "mongo_record": mongo_ok,
            "minio_object": minio_ok,
        },
    }


def _delete_milvus(file_title: str) -> int:
    """删除 kb_chunks_v1 中 file_title 对应的所有切片，返回删除数量"""
    if not file_title:
        logger.warning("[delete] file_title 为空，跳过 Milvus chunks 删除")
        return 0
    try:
        milvus_client = StorageClients.get_milvus_client()
        collection_name = os.getenv("CHUNKS_COLLECTION", "kb_chunks_v1")
        count = milvus_util.delete_by_file_title(milvus_client, collection_name, file_title)
        if count > 0:
            logger.info(f"[delete] 已从 {collection_name} 删除 file_title={file_title}，共 {count} 条切片")
        elif count == 0:
            logger.info(f"[delete] {collection_name} 中无匹配记录: file_title={file_title}")
        return count
    except Exception as e:
        logger.error(f"[delete] Milvus chunks 删除异常 file_title={file_title}: {e}")
        return -1


def _delete_item_names(file_title: str) -> int:
    """删除 kb_item_names_v1 中 file_title 对应的商品名记录，返回删除数量"""
    if not file_title:
        return 0
    try:
        milvus_client = StorageClients.get_milvus_client()
        collection_name = os.getenv("ITEM_NAME_COLLECTION", "kb_item_names_v1")
        if not milvus_client.has_collection(collection_name):
            logger.warning(f"[delete] 集合 {collection_name} 不存在，跳过")
            return 0
        result = milvus_client.delete(
            collection_name=collection_name,
            filter=f'file_title == "{file_title}"',
        )
        count = result.get("delete_count", 0) if isinstance(result, dict) else 0
        if count > 0:
            logger.info(f"[delete] 已从 {collection_name} 删除 file_title={file_title}，共 {count} 条商品名记录")
        else:
            logger.info(f"[delete] {collection_name} 中无匹配记录: file_title={file_title}")
        return count
    except Exception as e:
        logger.error(f"[delete] Milvus item_names 删除异常 file_title={file_title}: {e}")
        return -1


def _delete_minio(object_path: str) -> bool:
    if not object_path:
        return True
    try:
        minio_client = StorageClients.get_minio_client()
        bucket = mongo_import_util.get_bucket_name()
        minio_client.remove_object(bucket, object_path)
        logger.info(f"[delete] MinIO 已删除: {bucket}/{object_path}")
        return True
    except Exception as e:
        logger.error(f"[delete] MinIO 删除异常: {e}")
        return False