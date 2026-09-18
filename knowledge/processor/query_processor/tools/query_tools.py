import json
from typing import Any, Dict, List

from knowledge.processor.query_processor.tools.base import AgentTool, ToolRegistry


class KnowledgeBaseTool(AgentTool):
    """从已检索+重排序后的知识库文档中获取内容（首要信息来源，必须优先调用）。"""

    name = "search_knowledge_base"
    description = (
        "【首要工具，必须优先调用】从本地知识库中获取产品的权威技术文档、使用说明、规格参数等内容。"
        "知识库由产品PDF手册导入，内容具有最高优先级。"
        "仅当本工具返回'知识库中暂无相关文档'时，才允许调用 search_web。"
    )
    parameters = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "你想了解的具体技术问题或关键词",
            },
            "top_k": {
                "type": "integer",
                "description": "返回多少条相关结果（默认3，最多5）",
                "default": 3,
            },
        },
        "required": ["query"],
    }

    def __init__(self, reranked_docs: List[Dict[str, Any]], item_names: List[str]):
        self._docs = reranked_docs or []
        self._item_names = item_names or []

    def run(self, query: str = "", top_k: int = 3) -> str:
        top_k = max(1, min(top_k, 5))
        if not self._docs:
            return "知识库中暂无相关文档。"

        results = []
        for i, doc in enumerate(self._docs[:top_k]):
            content = doc.get("content", "")
            title = doc.get("title", "")
            source = doc.get("source", "")
            score = doc.get("score")
            score_str = f" (相关度:{score:.3f})" if score is not None else ""
            results.append(
                f"[文档{i + 1}] 标题:{title} 来源:{source}{score_str}\n{content}"
            )

        header = f"知识库搜索结果（产品:{'、'.join(self._item_names) if self._item_names else '未知'}）:"
        return header + "\n\n" + "\n\n".join(results)


class WebSearchTool(AgentTool):
    """从互联网搜索实时信息（仅当知识库无相关内容时才允许使用）。"""

    name = "search_web"
    description = (
        "【辅助工具，仅在知识库无结果时使用】从互联网搜索实时信息，包括产品最新价格、库存状态、用户评价、行业新闻等。"
        "注意：本工具为知识库的补充手段，搜索结果优先级低于知识库内容。"
        "仅当 search_knowledge_base 返回'知识库中暂无相关文档'时才应调用本工具。"
    )
    parameters = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "搜索关键词，建议包含产品名称和你想了解的具体信息",
            },
            "count": {
                "type": "integer",
                "description": "返回多少条搜索结果（默认3）",
                "default": 3,
            },
        },
        "required": ["query"],
    }

    def __init__(self, web_docs: List[Dict[str, Any]]):
        self._docs = web_docs or []

    def run(self, query: str = "", count: int = 3) -> str:
        if not self._docs:
            return "本次查询未触发联网搜索，或联网搜索服务不可用。"

        count = max(1, min(count, 5))
        results = []
        for i, doc in enumerate(self._docs[:count]):
            title = doc.get("title", "")
            snippet = doc.get("snippet", "")
            url = doc.get("url", "")
            results.append(f"[结果{i + 1}] {title}\n{snippet}\n来源: {url}")

        return "联网搜索结果:\n\n" + "\n\n".join(results)


class RealTimePriceTool(AgentTool):
    """联网查询产品价格（仅当知识库无价格信息时才使用）。"""

    name = "get_realtime_price"
    description = (
        "【辅助工具，仅在知识库无价格信息时使用】联网查询产品的最新市场价格、批量折扣、库存状态和购买渠道。"
        "注意：价格信息以网络搜索结果为准（知识库PDF通常不含实时价格），但产品规格参数必须以知识库为准。"
    )
    parameters = {
        "type": "object",
        "properties": {
            "product_name": {
                "type": "string",
                "description": "完整的产品名称，如'RS PRO RS-12 数字万用表'",
            },
        },
        "required": ["product_name"],
    }

    def __init__(self, web_docs: List[Dict[str, Any]]):
        self._docs = web_docs or []

    def run(self, product_name: str = "") -> str:
        if not self._docs:
            return "暂无法获取实时价格信息（联网搜索未触发或服务不可用）。"

        price_lines = []
        for doc in self._docs:
            snippet = doc.get("snippet", "")
            title = doc.get("title", "")
            url = doc.get("url", "")
            if any(kw in (snippet + title) for kw in ["价", "¥", "$", "￥", "元", "折扣", "库存", "现货", "购买"]):
                price_lines.append(f"- {title}: {snippet[:200]}  (来源: {url})")

        if not price_lines:
            all_lines = [
                f"- {d.get('title', '')}: {d.get('snippet', '')[:200]}  (来源: {d.get('url', '')})"
                for d in self._docs
            ]
            return (
                f"关于「{product_name}」的实时价格信息暂未获取到，但有以下联网搜索结果可参考:\n"
                + "\n".join(all_lines)
            )

        return f"「{product_name}」的实时价格/购买信息:\n" + "\n".join(price_lines)


class CompareProductsTool(AgentTool):
    """对比两款产品的参数。"""

    name = "compare_products"
    description = "对比两款相似产品的技术参数、功能差异、适用场景等。适合回答'A和B有什么区别'、'选A还是B'等问题。"
    parameters = {
        "type": "object",
        "properties": {
            "product_a": {
                "type": "string",
                "description": "第一款产品名称",
            },
            "product_b": {
                "type": "string",
                "description": "第二款产品名称",
            },
            "aspects": {
                "type": "array",
                "items": {"type": "string"},
                "description": "重点对比的方面，如['规格参数', '价格', '适用场景']",
                "default": ["规格参数", "价格", "适用场景"],
            },
        },
        "required": ["product_a", "product_b"],
    }

    def __init__(self, reranked_docs: List[Dict[str, Any]], web_docs: List[Dict[str, Any]]):
        self._docs = reranked_docs or []
        self._web = web_docs or []

    def run(self, product_a: str = "", product_b: str = "", aspects: List[str] = None) -> str:
        aspects = aspects or ["规格参数", "价格", "适用场景"]

        relevant_docs = []
        for doc in self._docs:
            content = doc.get("content", "")
            if product_a in content or product_b in content:
                relevant_docs.append(content[:300])

        relevant_web = []
        for doc in self._web:
            text = doc.get("title", "") + " " + doc.get("snippet", "")
            if product_a in text or product_b in text:
                relevant_web.append(text[:200])

        return (
            f"请从以下资料中对比「{product_a}」和「{product_b}」在 {aspects} 方面的差异:\n\n"
            f"=== 知识库资料 ===\n"
            + ("\n".join(f"- {d}" for d in relevant_docs) if relevant_docs else "(无相关知识库文档)")
            + f"\n\n=== 联网资料 ===\n"
            + ("\n".join(f"- {w}" for w in relevant_web) if relevant_web else "(无相关联网资料)")
        )


def build_tools(
    reranked_docs: List[Dict[str, Any]],
    web_docs: List[Dict[str, Any]],
    item_names: List[str],
) -> List[AgentTool]:
    """根据 state 中的数据构建工具列表。"""

    import os
    import logging
    _log = logging.getLogger(__name__)

    ToolRegistry.clear()

    tools = [
        KnowledgeBaseTool(reranked_docs=reranked_docs, item_names=item_names),
    ]

    mcp_web_search_configured = bool(os.getenv("MCP_DASHSCOPE_BASE_URL", "").strip())
    has_web_docs = bool(web_docs)

    if mcp_web_search_configured:
        tools.append(WebSearchTool(web_docs=web_docs))
        tools.append(RealTimePriceTool(web_docs=web_docs))
        _log.info(
            f"[Agent Tool] 联网搜索已配置 MCP_DASHSCOPE_BASE_URL，"
            f"启用 WebSearchTool + RealTimePriceTool (web_docs={len(web_docs)}条)"
        )
    else:
        _log.warning(
            "[Agent Tool] 未配置 MCP_DASHSCOPE_BASE_URL，自动禁用 WebSearchTool 和 RealTimePriceTool，避免 Agent 报错"
        )

    tools.append(CompareProductsTool(reranked_docs=reranked_docs, web_docs=web_docs))

    for t in tools:
        ToolRegistry.register(t)

    _log.info(f"[Agent Tool] 共启用 {len(tools)} 个工具: {[t.name for t in tools]}")
    return tools