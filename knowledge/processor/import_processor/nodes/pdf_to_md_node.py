import os
import sys
import json
import time
import logging
import subprocess
from pathlib import Path
from typing import Optional

import requests

from knowledge.processor.import_processor.base import BaseNode
from knowledge.processor.import_processor.state import ImportGraphState
from knowledge.processor.import_processor.exceptions import (
    StateFieldError, FileProcessingError,
)

logger = logging.getLogger(__name__)

MINERU_HOST = "127.0.0.1"
MINERU_PORT = 61011
MINERU_URL = f"http://{MINERU_HOST}:{MINERU_PORT}"
MINERU_STARTUP_TIMEOUT = 120
MINERU_PARSE_TIMEOUT = 600

_mineru_process: Optional[subprocess.Popen] = None


def _ensure_mineru_running() -> None:
    global _mineru_process

    try:
        r = requests.get(f"{MINERU_URL}/health", timeout=5)
        if r.status_code == 200:
            return
    except Exception:
        pass

    env = os.environ.copy()
    env.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    env.setdefault("CUDA_VISIBLE_DEVICES", "")
    env.setdefault("MINERU_DEVICE_MODE", "cpu")

    _mineru_process = subprocess.Popen(
        [sys.executable, "-m", "mineru.cli.fast_api",
         "--host", MINERU_HOST, "--port", str(MINERU_PORT)],
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    deadline = time.time() + MINERU_STARTUP_TIMEOUT
    while time.time() < deadline:
        try:
            r = requests.get(f"{MINERU_URL}/health", timeout=5)
            if r.status_code == 200:
                logger.info("MinerU API ready (pid=%s)", _mineru_process.pid)
                return
        except Exception:
            pass
        if _mineru_process.poll() is not None:
            raise RuntimeError(
                f"MinerU API exited prematurely (code={_mineru_process.returncode})"
            )
        time.sleep(2)

    raise TimeoutError(f"MinerU API not healthy within {MINERU_STARTUP_TIMEOUT}s")


def _call_mineru_parse(pdf_path: Path) -> str:
    _ensure_mineru_running()

    filename = pdf_path.name
    logger.info("Calling MinerU API to parse %s ...", filename)

    with open(pdf_path, "rb") as f:
        r = requests.post(
            f"{MINERU_URL}/file_parse",
            files={"files": (filename, f, "application/pdf")},
            data={
                "backend": "pipeline",
                "parse_method": "txt",
                "lang_list": ["ch"],
            },
            timeout=MINERU_PARSE_TIMEOUT,
        )

    if r.status_code != 200:
        raise RuntimeError(f"MinerU API error {r.status_code}: {r.text[:500]}")

    data = r.json()
    results = data.get("results", {})
    if not results:
        raise RuntimeError(
            f"MinerU returned no results: {json.dumps(data, ensure_ascii=False)[:500]}"
        )

    first_key = next(iter(results))
    md_content = results[first_key].get("md_content", "")
    if not md_content:
        raise RuntimeError(f"MinerU returned empty md_content for {first_key}")

    logger.info("MinerU parsed OK, md length=%d", len(md_content))
    return md_content


class PdfToMdNode(BaseNode):
    name = "pdf_to_md_node"

    def __init__(self, config=None):
        super().__init__(config)
        _ensure_mineru_running()

    def process(self, state: ImportGraphState) -> ImportGraphState:
        pdf_path = state.get('pdf_path', '')
        file_dir = state.get('file_dir', '')
        file_title = state.get('file_title', '')

        if not pdf_path:
            raise StateFieldError(node_name=self.name, field_name='pdf_path', expected_type=str)
        if not file_dir:
            raise StateFieldError(node_name=self.name, field_name='file_dir', expected_type=str)

        pdf_path_obj = Path(pdf_path)
        file_dir_obj = Path(file_dir)

        if not pdf_path_obj.exists():
            raise StateFieldError(node_name=self.name, field_name='pdf_path', expected_type=Path)
        if not file_dir_obj.exists():
            raise StateFieldError(node_name=self.name, field_name='file_dir', expected_type=Path)

        md_file = file_dir_obj / f"{file_title}.md"

        self.log_step("Parse", f"MinerU pipeline -> {pdf_path_obj.name}")

        try:
            md_content = _call_mineru_parse(pdf_path_obj)
        except Exception as e:
            self.logger.error("MinerU parse failed: %s", e)
            raise FileProcessingError(
                message=f"PDF解析失败: {e}", node_name=self.name
            )

        md_file.write_text(md_content, encoding="utf-8")
        self.log_step("Save", f"Markdown saved -> {md_file} ({len(md_content)} chars)")

        state['md_path'] = str(md_file)
        state['md_content'] = md_content

        return state