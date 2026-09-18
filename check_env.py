"""校验 .env 合并后环境变量加载是否正常"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv

load_dotenv()

KEYS = [
    "OPENAI_API_KEY",
    "OPENAI_API_BASE",
    "LLM_DEFAULT_MODEL",
    "BGE_DEVICE",
    "BGE_FP16",
    "BGE_RERANKER_DEVICE",
    "EMBEDDING_DIM",
    "EMBEDDING_MODEL",
    "MILVUS_URL",
    "CHUNKS_COLLECTION",
    "ITEM_NAME_COLLECTION",
    "MILVUS_METRIC_TYPE",
    "MILVUS_MIN_COSINE_SCORE",
    "MONGO_URL",
    "MONGO_DB_NAME",
    "MINIO_ENDPOINT",
    "MINIO_BUCKET_NAME",
    "HF_ENDPOINT",
    "MINERU_MODEL_DIR",
    "MINERU_DEVICE_MODE",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "ENABLE_AGENT_MODE",
    "AGENT_MAX_TURNS",
]

print("=" * 55)
print("  .env 环境变量加载校验")
print("=" * 55)

missing = []
for k in KEYS:
    v = os.getenv(k)
    if v is None:
        status = "❌ 缺失"
        missing.append(k)
    else:
        masked = v
        if k in ("OPENAI_API_KEY", "MONGO_URL", "MINIO_ACCESS_KEY", "MINIO_SECRET_KEY"):
            masked = v[:12] + "***" if len(v) > 12 else "***"
        status = masked
    print(f"  {k:<30s} = {status}")

print("=" * 55)
if missing:
    print(f"  ❌ 缺少 {len(missing)} 个变量: {missing}")
    sys.exit(1)
else:
    print(f"  ✅ 全部 {len(KEYS)} 个变量加载正常")
    sys.exit(0)