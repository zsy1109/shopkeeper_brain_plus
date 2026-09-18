import hashlib
import os
import os.path
import logging
import shutil
import time
import uuid
from datetime import datetime
from fastapi import UploadFile
from pypdf import PdfReader
from pypdf.errors import PyPdfError
from knowledge.core.paths import get_local_base_dir
from knowledge.processor.import_processor.exceptions import FileProcessingError, FileValidationError
from knowledge.utils.client.storage_clients import StorageClients
from knowledge.processor.import_processor.main_graph import get_import_graph
from knowledge.utils.task_util import update_task_status, add_running_task, add_done_task, add_node_duration, \
    TASK_STATUS_PROCESSING, \
    TASK_STATUS_COMPLETED, \
    TASK_STATUS_FAILED
from knowledge.utils import mongo_import_util
from knowledge.utils import milvus_util

logger = logging.getLogger(__name__)


class UpLoadService:

    def get_base_dir(self) -> str:
        # %Y%m%d:年月日
        # %Y:四位 %y:两位
        return os.path.join(get_local_base_dir(), datetime.now().strftime("%Y%m%d"))

    """
    处理文件上传相关的逻辑
    """

    def run_import_graph(self, task_id: str, import_file_path: str, file_dir: str, minio_object_path: str = "", file_md5: str = ""):
        """
        运行整个图谱流程
        Args:
            task_id:
            import_file_path:
            file_dir:
            minio_object_path:
            file_md5:

        Returns:

        """

        update_task_status(task_id, TASK_STATUS_PROCESSING)

        graph_state = {
            "task_id": task_id,
            "import_file_path": import_file_path,
            "file_dir": file_dir
        }

        final_state: dict = {}
        try:
            for event in get_import_graph().stream(graph_state):
                for key, value in event.items():
                    logger.info(f"当前正在执行的节点--->{key}")
                    final_state = value

            update_task_status(task_id, TASK_STATUS_COMPLETED)
            self._write_import_record(
                task_id=task_id,
                import_file_path=import_file_path,
                minio_object_path=minio_object_path,
                final_state=final_state,
                file_md5=file_md5,
            )
        except Exception as e:
            logger.error(f"[{task_id}] 执行导入过程中出现异常 原因:{str(e)}")
            update_task_status(task_id, TASK_STATUS_FAILED)

    def _write_import_record(self,
                             task_id: str,
                             import_file_path: str,
                             minio_object_path: str,
                             final_state: dict,
                             file_md5: str = ""):
        filename = os.path.basename(import_file_path)
        file_title = final_state.get("file_title", filename.rsplit(".", 1)[0] if "." in filename else filename)
        item_name = final_state.get("item_name", "")
        chunks = final_state.get("chunks", [])
        chunk_count = len(chunks) if isinstance(chunks, list) else 0

        mongo_import_util.create_import_record(
            task_id=task_id,
            filename=filename,
            file_title=file_title,
            item_name=item_name,
            chunk_count=chunk_count,
            status="completed",
            minio_object_path=minio_object_path,
            import_file_path=import_file_path,
            file_md5=file_md5,
        )

    def process_upload_file(self, file: UploadFile):
        """
        处理文件上传

        1. 校验文件后缀（仅 .pdf）
        2. 校验文件大小（0字节拦截）
        3. 将上传的文件存储到本地临时目录(主要为了做中转)，同时计算MD5
        4. PDF 完整性校验（损坏/加密拦截）
        5. 根据文件MD5查重，重复则拦截
        6. 将上传的文件存储到远程minio(主要持久化)
        7. 将file_dir / import_file_path / task_id / md5 返回
        Args:
            file:

        Returns:

        """

        # ── 前置校验1：文件后缀 ──
        filename = file.filename or ""
        if not filename.lower().endswith('.pdf'):
            logger.warning(
                f"[upload] 后缀拦截 | 文件名='{filename}' | "
                f"原因: 仅支持 .pdf 格式 | 时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            )
            raise FileValidationError(
                message=f"只允许上传 PDF 文件，当前文件格式不支持 (文件名: {filename})"
            )

        # ── 前置校验2：空文件（0 字节）──
        content = file.file.read()
        if not content:
            logger.warning(
                f"[upload] 空文件拦截 | 文件名='{filename}' | "
                f"原因: 文件内容为空（0 字节） | 时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            )
            raise FileValidationError(
                message=f"文件内容为空（0 字节），请确认文件有效后重新上传 (文件名: {filename})"
            )
        file_size = len(content)
        file.file.seek(0)  # 重置指针，后续 save_upload_file_to_local 仍可正常读取

        # 1. 生成任务id
        task_id = str(uuid.uuid4().hex[:8])
        add_running_task(task_id, "upload_file")
        start_time = time.time()

        # 2. 生成日期目录并且将日期目录和临时目录拼接到一起
        base_file_dir = self.get_base_dir()

        # 3. 构建文档完整归属目录
        file_dir = os.path.join(base_file_dir, task_id)

        # 4. 保存文件到临时目录（同时计算MD5）
        import_file_path, md5_hash = self.save_upload_file_to_local(file, file_dir)

        # ── 前置校验3：PDF 完整性（损坏 / 加密）──
        self._validate_pdf_integrity(import_file_path, filename)

        # 5. 查重：根据文件MD5查询 MongoDB
        duplicate = mongo_import_util.find_duplicate_by_md5(md5_hash)
        if duplicate:
            shutil.rmtree(file_dir, ignore_errors=True)
            logger.warning(
                f"[upload] 重复文件拦截 | 文件名='{filename}' | MD5={md5_hash} | "
                f"已存在_id={duplicate.get('_id')} | 时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            )
            raise FileProcessingError(message=f"该文档已上传至知识库，请勿重复上传")

        # 6. 保存文件到minio中
        minio_object_path = self.save_upload_file_to_minio(import_file_path, file.filename)
        end_time = time.time()
        add_done_task(task_id, "upload_file")
        add_node_duration(task_id, "upload_file", end_time - start_time)

        # 7. 返回图谱的信息
        return task_id, import_file_path, file_dir, minio_object_path, md5_hash

    @staticmethod
    def _validate_pdf_integrity(file_path: str, filename: str):
        """校验 PDF 文件完整性：检测损坏或加密的 PDF

        Args:
            file_path: 本地 PDF 文件路径
            filename: 原始文件名（用于日志）

        Raises:
            FileValidationError: PDF 损坏或加密时抛出
        """
        try:
            reader = PdfReader(file_path)
            if reader.is_encrypted:
                logger.warning(
                    f"[upload] 加密PDF拦截 | 文件名='{filename}' | "
                    f"原因: PDF 已加密，无法解析 | 时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
                )
                raise FileValidationError(
                    message=f"PDF 文件已加密，无法解析内容，请上传未加密的 PDF 文件 (文件名: {filename})"
                )
            page_count = len(reader.pages)
            if page_count == 0:
                logger.warning(
                    f"[upload] 空白PDF拦截 | 文件名='{filename}' | "
                    f"原因: PDF 无页面内容 | 时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
                )
                raise FileValidationError(
                    message=f"PDF 文件无页面内容，请确认文件有效后重新上传 (文件名: {filename})"
                )
            logger.info(f"[upload] PDF 完整性校验通过: {filename}, 页数={page_count}")
        except FileValidationError:
            raise
        except PyPdfError as e:
            logger.warning(
                f"[upload] 损坏PDF拦截 | 文件名='{filename}' | "
                f"原因: PDF 解析失败 | 时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            )
            raise FileValidationError(
                message=f"PDF 文件已损坏，无法正常打开，请重新生成 PDF 后上传 (文件名: {filename})"
            ) from e
        except Exception as e:
            logger.warning(
                f"[upload] PDF校验异常 | 文件名='{filename}' | "
                f"原因: {e} | 时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            )
            raise FileValidationError(
                message=f"PDF 文件无法正常读取，请确认文件有效性 (文件名: {filename})"
            ) from e

    def save_upload_file_to_local(self, file: UploadFile, file_dir: str):
        """
        保存文件到临时目录，同时计算 MD5

        Args:
            file: 文件上传对象
            file_dir: 上传文件的目录

        Returns:
            (import_file_path, md5_hash)

        """
        # 1. 创建文件的归属目录
        os.makedirs(file_dir, exist_ok=True)

        # 2. 构建导入文件的路径
        import_file_path = os.path.join(file_dir, file.filename)

        # 3. 边写边算 MD5
        md5 = hashlib.md5()
        try:
            with open(import_file_path, "wb") as f:
                while True:
                    chunk = file.file.read(1024 * 1024)
                    if not chunk:
                        break
                    f.write(chunk)
                    md5.update(chunk)
        except IOError as e:
            logger.info(f"{file.filename}写入临时目录失败 原因:{str(e)}")
            raise FileProcessingError(message=f"{file.filename}写入临时目录失败 原因:{str(e)}")

        md5_hash = md5.hexdigest()
        logger.info(f"[upload] {file.filename} 写入完成, MD5={md5_hash}")

        # 4. 返回导入的文件路径和md5
        return import_file_path, md5_hash

    def save_upload_file_to_minio(self, import_file_path: str, filename: str) -> str:
        """

        Args:
            import_file_path:  上传文件的地址
            filename: 上传文件的名字

        Returns:
            minio_object_path 上传到 MinIO 的对象路径

        """

        # 1. 获取minio客户端
        try:
            minio_client = StorageClients.get_minio_client()
        except ConnectionError as e:
            logger.error(f"MinIO客户端获取失败 原因:{str(e)}")
            return ""

        # 2. 获取minio相关信息
        bucket_name = os.getenv('MINIO_BUCKET_NAME')
        object_name = f"origin_files/{datetime.now().strftime('%Y%m%d')}/{filename}"

        # 3. 上传
        try:
            minio_client.fput_object(bucket_name, object_name, import_file_path)
            return object_name
        except Exception as e:
            logger.error(f"{filename}上传到MinIO失败 原因：{str(e)}")
            return ""