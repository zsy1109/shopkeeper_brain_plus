"""三端数据巡检脚本：MongoDB + kb_item_names_v1 + kb_chunks_v1"""
import sys

import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pymilvus import MilvusClient
from dotenv import load_dotenv
from knowledge.utils.client.storage_clients import StorageClients

load_dotenv()


def check_mongo():
    print("=" * 50)
    print("[MongoDB] import_record")
    print("=" * 50)
    db = StorageClients.get_mongo_db()
    recs = list(db.import_record.find({}))
    print(f"  记录数: {len(recs)}")
    if recs:
        for r in recs:
            print(
                f"  file_id={r.get('_id')}, "
                f"filename={r.get('filename', '')}, "
                f"file_title={r.get('file_title', '')}, "
                f"item_name={r.get('item_name', '')}"
            )
    else:
        print("  (空)")
    print()
    return recs


def check_milvus(collection_name, output_fields):
    print("=" * 50)
    print(f"[Milvus] {collection_name}")
    print("=" * 50)
    c = MilvusClient(uri="http://192.168.40.140:19530")
    res = c.query(
        collection_name=collection_name,
        filter="",
        output_fields=output_fields,
        limit=100,
    )
    print(f"  记录数: {len(res)}")
    if res:
        id_field = output_fields[0] if output_fields else "pk"
        for r in res:
            parts = [f"{id_field}={r[id_field]}"]
            for f in output_fields:
                if f != id_field:
                    parts.append(f"{f}={r.get(f, '')}")
            print("  " + ", ".join(parts))
    else:
        print("  (空)")
    print()
    return res


def main():
    print()

    # 1. MongoDB
    mongo_recs = check_mongo()

    # 2. kb_item_names_v1
    item_names = check_milvus("kb_item_names_v1", ["pk", "item_name", "file_title"])

    # 3. kb_chunks_v1
    chunks = check_milvus("kb_chunks_v1", ["chunk_id", "file_title", "title", "item_name"])

    # 4. 按 file_title 聚合 chunks
    print("=" * 50)
    print("[汇总] kb_chunks_v1 按 file_title 分布")
    print("=" * 50)
    if chunks:
        titles = {}
        for r in chunks:
            ft = r.get("file_title", "")
            titles[ft] = titles.get(ft, 0) + 1
        for k, v in titles.items():
            print(f"  {k}: {v} chunks")
    else:
        print("  (空)")
    print()

    # 5. 校验 AI应用开发实习生
    print("=" * 50)
    print("[校验] AI应用开发实习生 是否清除干净")
    print("=" * 50)
    ai_in_mongo = any("AI应用开发实习生" in str(r) for r in mongo_recs)
    ai_in_names = any("AI应用开发实习生" in str(r) for r in item_names)
    ai_in_chunks = any("AI应用开发实习生" in str(r) for r in chunks)

    all_clean = not ai_in_mongo and not ai_in_names and not ai_in_chunks
    print(f"  MongoDB:          {'残留' if ai_in_mongo else '已清除'}")
    print(f"  kb_item_names_v1: {'残留' if ai_in_names else '已清除'}")
    print(f"  kb_chunks_v1:     {'残留' if ai_in_chunks else '已清除'}")
    print(f"  总判定: {'✅ 全部清除' if all_clean else '❌ 仍有残留'}")
    print()

    return 0 if all_clean else 1


if __name__ == "__main__":
    sys.exit(main())