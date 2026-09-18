"""
孤儿数据清理脚本 — 保障 MongoDB 与 Milvus 双向数据一致性

用法:
    # 预览模式（只打印脏数据清单，不执行删除）
    python clean_orphan_data.py --dry-run

    # 执行模式（确认清理脏数据）
    python clean_orphan_data.py --execute

检测逻辑:
    1. 孤儿向量：Milvus kb_chunks_v1 / kb_item_names_v1 中存在，
       但 MongoDB import_record 中已无对应 file_title 的记录
    2. 孤儿元数据：MongoDB import_record 中存在记录，
       但 Milvus kb_chunks_v1 中无对应向量切片（或 kb_item_names_v1 中无商品名）

日志输出: 同时输出到控制台和 clean_orphan_data.log 文件
"""

import argparse
import logging
import os
import sys
from datetime import datetime
from typing import Dict, List, Set, Tuple

from dotenv import load_dotenv

load_dotenv()

from knowledge.utils.client.storage_clients import StorageClients
from knowledge.utils import milvus_util
from knowledge.utils import mongo_import_util

# ── 日志配置 ──
LOG_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "clean_orphan_data.log")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
    ],
)
logger = logging.getLogger(__name__)

CHUNKS_COLLECTION = os.getenv("CHUNKS_COLLECTION", "kb_chunks_v1")
ITEM_NAMES_COLLECTION = os.getenv("ITEM_NAME_COLLECTION", "kb_item_names_v1")


# ═══════════════════════════════════════════════════════════════
#  数据采集
# ═══════════════════════════════════════════════════════════════

def _collect_mongo_file_titles() -> Dict[str, dict]:
    """获取 MongoDB import_record 中所有记录的 file_title → 摘要映射。"""
    records = mongo_import_util.list_import_records(limit=5000)
    result = {}
    for r in records:
        ft = (r.get("file_title") or "").strip()
        if ft:
            result[ft] = {
                "file_id": r.get("file_id", ""),
                "filename": r.get("filename", ""),
                "item_name": r.get("item_name", ""),
                "chunk_count": r.get("chunk_count", 0),
                "status": r.get("status", ""),
            }
    return result


def _collect_milvus_file_titles(collection_name: str) -> Set[str]:
    """获取 Milvus 某个 collection 中所有不重复的 file_title。"""
    try:
        client = StorageClients.get_milvus_client()
        results = client.query(
            collection_name=collection_name,
            filter="",
            output_fields=["file_title"],
            limit=10000,
        )
        titles = set()
        for r in (results or []):
            ft = (r.get("file_title") or "").strip()
            if ft:
                titles.add(ft)
        return titles
    except Exception as e:
        logger.error(f"[Milvus] 查询 {collection_name} 失败: {e}")
        return set()


def _collect_milvus_detail(collection_name: str) -> Dict[str, int]:
    """获取 Milvus 某个 collection 中每个 file_title 的记录数。"""
    try:
        client = StorageClients.get_milvus_client()
        results = client.query(
            collection_name=collection_name,
            filter="",
            output_fields=["file_title"],
            limit=10000,
        )
        counts: Dict[str, int] = {}
        for r in (results or []):
            ft = (r.get("file_title") or "").strip()
            if ft:
                counts[ft] = counts.get(ft, 0) + 1
        return counts
    except Exception as e:
        logger.error(f"[Milvus] 查询 {collection_name} 详情失败: {e}")
        return {}


# ═══════════════════════════════════════════════════════════════
#  脏数据检测
# ═══════════════════════════════════════════════════════════════

def detect_orphan_vectors(
    mongo_titles: Dict[str, dict],
    chunks_titles: Set[str],
    item_names_titles: Set[str],
    chunks_detail: Dict[str, int],
    names_detail: Dict[str, int],
) -> List[dict]:
    """检测孤儿向量：Milvus 有但 MongoDB 没有的 file_title。"""
    orphans = []
    all_milvus_titles = chunks_titles | item_names_titles
    for ft in sorted(all_milvus_titles):
        if ft not in mongo_titles:
            orphans.append({
                "file_title": ft,
                "chunks_count": chunks_detail.get(ft, 0),
                "item_names_count": names_detail.get(ft, 0),
            })
    return orphans


def detect_orphan_metadata(
    mongo_titles: Dict[str, dict],
    chunks_detail: Dict[str, int],
    names_detail: Dict[str, int],
) -> dict:
    """检测孤儿元数据：MongoDB 有但 Milvus 数据不全的 file_title。"""
    missing_chunks: List[dict] = []
    missing_names: List[dict] = []

    for ft, info in sorted(mongo_titles.items()):
        has_chunks = chunks_detail.get(ft, 0) > 0
        has_names = names_detail.get(ft, 0) > 0

        if not has_chunks:
            missing_chunks.append({
                "file_id": info["file_id"],
                "file_title": ft,
                "filename": info["filename"],
                "item_name": info["item_name"],
                "expected_chunks": info["chunk_count"],
                "reason": "kb_chunks_v1 中无对应切片",
            })

        if not has_names:
            missing_names.append({
                "file_id": info["file_id"],
                "file_title": ft,
                "filename": info["filename"],
                "item_name": info["item_name"],
                "reason": "kb_item_names_v1 中无对应商品名",
            })

    return {"missing_chunks": missing_chunks, "missing_names": missing_names}


# ═══════════════════════════════════════════════════════════════
#  清理执行
# ═══════════════════════════════════════════════════════════════

def execute_clean_orphan_vectors(orphans: List[dict]) -> Tuple[int, int]:
    """删除 Milvus 中的孤儿向量（chunks + item_names）。"""
    client = StorageClients.get_milvus_client()
    total_chunks_deleted = 0
    total_names_deleted = 0

    for item in orphans:
        ft = item["file_title"]
        c_count = item["chunks_count"]
        n_count = item["item_names_count"]

        if c_count > 0:
            deleted = milvus_util.delete_by_file_title(client, CHUNKS_COLLECTION, ft)
            if deleted >= 0:
                total_chunks_deleted += deleted
                logger.info(f"[清理-向量] 已删除 kb_chunks_v1 中 file_title='{ft}' 的 {deleted} 条切片")
            else:
                logger.error(f"[清理-向量] 删除 kb_chunks_v1 中 file_title='{ft}' 失败")

        if n_count > 0:
            deleted = milvus_util.delete_by_file_title(client, ITEM_NAMES_COLLECTION, ft)
            if deleted >= 0:
                total_names_deleted += deleted
                logger.info(f"[清理-向量] 已删除 kb_item_names_v1 中 file_title='{ft}' 的 {deleted} 条记录")
            else:
                logger.error(f"[清理-向量] 删除 kb_item_names_v1 中 file_title='{ft}' 失败")

    return total_chunks_deleted, total_names_deleted


def execute_clean_orphan_metadata(orphan_metadata: dict) -> int:
    """删除 MongoDB 中缺少 Milvus 数据的孤儿记录。"""
    deleted_count = 0
    to_delete = set()
    for item in orphan_metadata.get("missing_chunks", []):
        to_delete.add(item["file_id"])
    for item in orphan_metadata.get("missing_names", []):
        to_delete.add(item["file_id"])

    for file_id in sorted(to_delete):
        ok = mongo_import_util.delete_import_record(file_id)
        if ok:
            deleted_count += 1
            logger.info(f"[清理-元数据] 已删除 MongoDB import_record 中 file_id={file_id}")
        else:
            logger.error(f"[清理-元数据] 删除 MongoDB import_record 中 file_id={file_id} 失败")

    return deleted_count


# ═══════════════════════════════════════════════════════════════
#  预览报告
# ═══════════════════════════════════════════════════════════════

def print_orphan_vector_report(orphans: List[dict]):
    if not orphans:
        logger.info("  ✓ 未发现孤儿向量（Milvus→MongoDB 方向一致）")
        return
    logger.warning(f"  ✗ 发现 {len(orphans)} 个孤儿向量（Milvus 中有但 MongoDB 中无对应记录）:")
    total_c = 0
    total_n = 0
    for item in orphans:
        total_c += item["chunks_count"]
        total_n += item["item_names_count"]
        logger.warning(
            f"    file_title='{item['file_title']}'  "
            f"chunks={item['chunks_count']}  item_names={item['item_names_count']}"
        )
    logger.warning(f"    合计: {total_c} 条切片 + {total_n} 条商品名 = {total_c + total_n} 条脏数据")


def print_orphan_metadata_report(orphan_metadata: dict):
    missing_chunks = orphan_metadata.get("missing_chunks", [])
    missing_names = orphan_metadata.get("missing_names", [])

    if not missing_chunks and not missing_names:
        logger.info("  ✓ 未发现孤儿元数据（MongoDB→Milvus 方向一致）")
        return

    if missing_chunks:
        logger.warning(f"  ✗ 发现 {len(missing_chunks)} 条孤儿元数据（MongoDB 有记录但 kb_chunks_v1 无切片）:")
        for item in missing_chunks:
            logger.warning(
                f"    file_id={item['file_id']}  file_title='{item['file_title']}'  "
                f"filename='{item['filename']}'  expected_chunks={item['expected_chunks']}"
            )

    if missing_names:
        logger.warning(f"  ✗ 发现 {len(missing_names)} 条孤儿元数据（MongoDB 有记录但 kb_item_names_v1 无商品名）:")
        for item in missing_names:
            logger.warning(
                f"    file_id={item['file_id']}  file_title='{item['file_title']}'  "
                f"filename='{item['filename']}'  item_name='{item['item_name']}'"
            )


def build_mongo_index():
    """构建 MongoDB file_title 索引（仅首次，幂等安全）。"""
    try:
        db = StorageClients.get_mongo_db()
        existing = db.import_record.index_information()
        if "file_title_1" not in existing:
            db.import_record.create_index("file_title", name="file_title_1")
            logger.info("[索引] MongoDB import_record.file_title 索引已创建")
    except Exception as e:
        logger.warning(f"[索引] 创建 MongoDB 索引失败（不影响主流程）: {e}")


# ═══════════════════════════════════════════════════════════════
#  主入口
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="MongoDB 与 Milvus 孤儿数据检测与清理")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--dry-run", action="store_true", help="预览模式：只打印脏数据清单，不执行删除")
    group.add_argument("--execute", action="store_true", help="执行模式：确认清理脏数据")
    args = parser.parse_args()

    is_dry_run = args.dry_run

    logger.info("=" * 60)
    logger.info(f"  孤儿数据清理脚本  |  模式: {'DRY-RUN 预览' if is_dry_run else 'EXECUTE 执行'}")
    logger.info(f"  时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"  日志文件: {LOG_FILE}")
    logger.info("=" * 60)

    # ── 建立 MongoDB 索引（加速后续查询） ──
    build_mongo_index()

    # ── 1. 采集数据 ──
    logger.info("\n[1/4] 采集 MongoDB 记录...")
    mongo_titles = _collect_mongo_file_titles()
    logger.info(f"  MongoDB import_record 中 file_title 总数: {len(mongo_titles)}")

    logger.info("\n[2/4] 采集 Milvus kb_chunks_v1 数据...")
    chunks_detail = _collect_milvus_detail(CHUNKS_COLLECTION)
    chunks_titles = set(chunks_detail.keys())
    logger.info(f"  kb_chunks_v1 中 file_title 总数: {len(chunks_titles)}, 切片总数: {sum(chunks_detail.values())}")

    logger.info("\n[3/4] 采集 Milvus kb_item_names_v1 数据...")
    names_detail = _collect_milvus_detail(ITEM_NAMES_COLLECTION)
    names_titles = set(names_detail.keys())
    logger.info(f"  kb_item_names_v1 中 file_title 总数: {len(names_titles)}, 记录总数: {sum(names_detail.values())}")

    # ── 4. 检测脏数据 ──
    logger.info("\n[4/4] 脏数据检测...")

    orphans = detect_orphan_vectors(mongo_titles, chunks_titles, names_titles, chunks_detail, names_detail)
    orphan_metadata = detect_orphan_metadata(mongo_titles, chunks_detail, names_detail)

    logger.info("\n" + "=" * 60)
    logger.info("  检测结果")
    logger.info("=" * 60)
    print_orphan_vector_report(orphans)
    print_orphan_metadata_report(orphan_metadata)

    total_issues = (
        len(orphans)
        + len(orphan_metadata.get("missing_chunks", []))
        + len(orphan_metadata.get("missing_names", []))
    )

    if total_issues == 0:
        logger.info("\n✅ MongoDB 与 Milvus 数据完全一致，无需清理")
        return 0

    if is_dry_run:
        logger.warning(f"\n⚠️ DRY-RUN 模式：以上为预览结果，共 {total_issues} 项脏数据，未执行实际删除")
        logger.info("  使用 --execute 参数执行实际清理")
        return 0

    # ── 5. 执行清理 ──
    logger.info("\n" + "=" * 60)
    logger.info("  执行清理")
    logger.info("=" * 60)

    total_chunks = 0
    total_names = 0
    total_mongo = 0

    if orphans:
        logger.info(f"\n→ 清理孤儿向量 ({len(orphans)} 个 file_title)...")
        tc, tn = execute_clean_orphan_vectors(orphans)
        total_chunks += tc
        total_names += tn

    if orphan_metadata.get("missing_chunks") or orphan_metadata.get("missing_names"):
        logger.info(f"\n→ 清理孤儿元数据...")
        total_mongo = execute_clean_orphan_metadata(orphan_metadata)

    # ── 6. 二次验证 ──
    logger.info("\n" + "=" * 60)
    logger.info("  二次验证（清理后重新检测）")
    logger.info("=" * 60)

    mongo_titles2 = _collect_mongo_file_titles()
    chunks_detail2 = _collect_milvus_detail(CHUNKS_COLLECTION)
    names_detail2 = _collect_milvus_detail(ITEM_NAMES_COLLECTION)
    chunks_titles2 = set(chunks_detail2.keys())
    names_titles2 = set(names_detail2.keys())

    orphans2 = detect_orphan_vectors(mongo_titles2, chunks_titles2, names_titles2, chunks_detail2, names_detail2)
    orphan_metadata2 = detect_orphan_metadata(mongo_titles2, chunks_detail2, names_detail2)

    remaining = (
        len(orphans2)
        + len(orphan_metadata2.get("missing_chunks", []))
        + len(orphan_metadata2.get("missing_names", []))
    )

    # ── 7. 汇总 ──
    logger.info("\n" + "=" * 60)
    logger.info("  清理汇总")
    logger.info("=" * 60)
    logger.info(f"  孤儿向量 - 删除切片:     {total_chunks} 条")
    logger.info(f"  孤儿向量 - 删除商品名:   {total_names} 条")
    logger.info(f"  孤儿元数据 - 删除MongoDB: {total_mongo} 条")
    logger.info(f"  清理后残留脏数据:       {remaining} 项")
    logger.info(f"  日志文件: {LOG_FILE}")

    if remaining == 0:
        logger.info("\n✅ 清理完成，MongoDB 与 Milvus 数据已完全一致")
    else:
        logger.warning(f"\n⚠️ 仍有 {remaining} 项脏数据未清理，请检查日志")

    return 0 if remaining == 0 else 1


if __name__ == "__main__":
    sys.exit(main())