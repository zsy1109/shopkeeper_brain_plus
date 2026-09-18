import os.path
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import uvicorn
from fastapi import FastAPI, UploadFile, Depends, BackgroundTasks, HTTPException, Query
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from knowledge.core.paths import get_front_page_dir
from knowledge.schema.upload_schema import UploadResponse,TaskStatusResponse
from knowledge.service.upload_service import UpLoadService
from knowledge.core.deps import get_upload_file_service
from knowledge.utils.task_util import get_task_info
from knowledge.utils import mongo_import_util
from knowledge.utils import milvus_util
from knowledge.utils.client.storage_clients import StorageClients
from knowledge.service import delete_service
from knowledge.processor.import_processor.exceptions import FileProcessingError, FileValidationError


class UTF8JSONResponse(JSONResponse):
    """自定义 JSONResponse，强制 media_type 携带 charset=utf-8"""
    media_type = "application/json; charset=utf-8"


# 1. 创建fastapi实例
# 2. 注册路由（将上传请求以及查询导入任务的请求注册到fastapi实例上）
# 3. 利用uvicorn服务器启动fastapi


def create_app():
    """
    创建FastAPI实例
    Returns:
    FastAPI实例

    """

    # 1. 实例化
    app = FastAPI(
        description="掌柜智库导入的应用",
        version="v1.0",
        default_response_class=UTF8JSONResponse,
    )

    # 2. 跨域配置
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # ← 和 credentials=True 冲突
        allow_credentials=False,  # 自定义cookies Authorization  tsl客户端证书信息
        allow_methods=["*"],  # ← 和 credentials=True 冲突  GET(获取资源)  POST(新增)  DELETE（删除） PUT（修改）
        allow_headers=["*"],  # ← 和 credentials=True 冲突   自定义的头字段 token  content-type:application/json
    )

    # 3. 挂载静态文件（import.html）
    page_dir = get_front_page_dir()

    if page_dir and os.path.exists(page_dir):
        app.mount("/front", StaticFiles(directory=page_dir))

    # 4. 注册路由
    register_router(app)

    # 5. 返回fastapi实例
    return app


def register_router(app: FastAPI):
    @app.get("/")
    def hello_world():
        return {"flag": "success"}

    # 1. 上传请求
    @app.post("/upload", response_model=UploadResponse)
    def upload_endpoint(file: UploadFile,
                        background_tasks: BackgroundTasks
                        , upload_service: UpLoadService = Depends(get_upload_file_service)):
        """
        处理文件的上传
        上传文件的原始名字：万用表的使用.pdf
        Returns:
        """
        # 1. 将上传的文件写入到本地临时目录以及远程MinIO（含MD5计算和查重）
        try:
            task_id, import_file_path, file_dir, minio_object_path, md5_hash = upload_service.process_upload_file(file)
        except FileValidationError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except FileProcessingError as e:
            raise HTTPException(status_code=409, detail=str(e))

        # 2. 运行整个导入的图谱(耗时：节点多【pdf解析很慢】)后台任务慢慢做
        background_tasks.add_task(
            upload_service.run_import_graph,
            task_id, import_file_path, file_dir, minio_object_path, md5_hash,
        )

        # 3. 返回上传后的响应（数据模型）
        return UploadResponse(message=f"{file.filename}文件上传成功", task_id=task_id)

    @app.get("/status/{task_id}")
    def get_task_status_endpoint(task_id: str):
        """
        查询上传任务：前端会轮训的调用查询上传任务状态结构。（1.5s轮训一次）
        # 轮询:1.性能 2.实时性
        # 1.5s--->实时性一般、性能相比极短时间轮询高很多。（节点很耗时）
        Returns:
        """

        task_info = get_task_info(task_id)

        return TaskStatusResponse(**task_info)

    # ── 知识库管理 ──

    @app.get("/files")
    def list_files_endpoint(limit: int = Query(default=200, ge=1, le=1000)):
        """获取已导入文件列表"""
        records = mongo_import_util.list_import_records(limit=limit)
        return {"total": len(records), "items": records}

    @app.get("/files/{file_id}")
    def get_file_endpoint(file_id: str):
        """获取单个文件的导入详情"""
        record = mongo_import_util.get_import_record(file_id)
        if not record:
            raise HTTPException(status_code=404, detail=f"文件记录不存在 (file_id={file_id})")
        return record

    @app.get("/files/{file_id}/chunks")
    def get_file_chunks_endpoint(file_id: str,
                                 limit: int = Query(default=200, ge=1, le=2000),
                                 offset: int = Query(default=0, ge=0)):
        """获取某文件在 Milvus 中的 chunks 预览"""
        record = mongo_import_util.get_import_record(file_id)
        if not record:
            raise HTTPException(status_code=404, detail=f"文件记录不存在 (file_id={file_id})")

        milvus_client = StorageClients.get_milvus_client()
        collection_name = os.getenv("CHUNKS_COLLECTION", "kb_chunks_v1")
        file_title = record.get("file_title", "")
        chunks = milvus_util.list_chunks_by_file_title(
            milvus_client, collection_name, file_title,
            limit=limit, offset=offset,
        )
        return {
            "file_title": file_title,
            "chunk_count": len(chunks),
            "items": chunks,
        }

    @app.delete("/files/{file_id}")
    def delete_file_endpoint(file_id: str):
        """删除已导入文档（Milvus + Mongo + MinIO 三层清理）"""
        result = delete_service.delete_document(file_id)
        if not result.get("success"):
            raise HTTPException(
                status_code=400,
                detail=result.get("message", "删除失败"),
            )
        return result





if __name__ == '__main__':
    # param1:fastapi实例
    # param2:启动的服务器地址
    # param3:启动的服务端口
    uvicorn.run(app=create_app(), host="0.0.0.0", port=8000, log_level="info")