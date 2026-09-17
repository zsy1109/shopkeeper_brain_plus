import json
from json import JSONDecodeError
from typing import Tuple, List, Dict, Any, Union

import httpx

from knowledge.processor.query_processor.base import BaseNode, T
from knowledge.processor.query_processor.state import QueryGraphState
from knowledge.processor.query_processor.exceptions import StateFieldError


class WebMcpSearchNode(BaseNode):
    name = "web_mcp_search_node"

    def process(self, state: QueryGraphState) -> Union[QueryGraphState, Dict[str, Any]]:
        rewritten_query, item_names = self._validate_state(state)

        try:
            web_search_results = self._execute_mcp_server(rewritten_query)
        except Exception as e:
            self.logger.warning(f"web_mcp_search_node 执行失败: {e}")
            return {}

        if not web_search_results:
            return {}

        return {"web_search_docs": web_search_results}

    def _validate_state(self, state: QueryGraphState) -> Tuple[str, List[str]]:
        rewritten_query = state.get('rewritten_query')
        item_names = state.get('item_names')

        if not rewritten_query or not isinstance(rewritten_query, str):
            raise StateFieldError(node_name=self.name, field_name='rewritten_query', expected_type=str)

        if not item_names or not isinstance(item_names, list):
            raise StateFieldError(node_name=self.name, field_name='item_names', expected_type=list)

        return rewritten_query, item_names

    def _execute_mcp_server(self, rewritten_query: str) -> List[Dict[str, Any]]:
        url = self.config.mcp_dashscope_base_url
        api_key = self.config.openai_api_key

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
        }

        payload = {
            "jsonrpc": "2.0",
            "method": "tools/call",
            "params": {
                "name": "bailian_web_search",
                "arguments": {"query": rewritten_query, "count": 3},
            },
            "id": 1,
        }

        with httpx.Client(timeout=30) as client:
            resp = client.post(url, headers=headers, json=payload)

        if resp.status_code != 200:
            self.logger.warning(f"web_mcp_search_node HTTP {resp.status_code}: {resp.text[:200]}")
            return []

        try:
            data = resp.json()
        except JSONDecodeError as e:
            self.logger.error(f"web_mcp_search_node 解析 JSON 失败: {e}")
            return []

        result = data.get("result", {})
        content_list = result.get("content", [])
        if not content_list:
            return []

        text_content = content_list[0].get("text", "") if isinstance(content_list[0], dict) else ""
        if not text_content:
            return []

        try:
            text_obj = json.loads(text_content)
        except JSONDecodeError as e:
            self.logger.error(f"web_mcp_search_node 解析搜索结果 JSON 失败: {e}")
            return []

        pages = text_obj.get("pages", [])
        return [
            {
                "snippet": page.get("snippet", "").strip(),
                "title": page.get("title", "").strip(),
                "url": page.get("url", "").strip(),
            }
            for page in pages
        ]


if __name__ == '__main__':
    web_search_node = WebMcpSearchNode()

    mock_state = {
        "rewritten_query": "RS-12 数字万用表如何测量直流电压？",
        "item_names": ["RS-12 数字万用表"],
    }

    web_search_node.process(mock_state)