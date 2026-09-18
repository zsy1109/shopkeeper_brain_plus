import json
import logging
from json import JSONDecodeError
from typing import Any, Callable, Dict, Generator, List, Optional, Tuple

import httpx

from knowledge.processor.query_processor.tools.base import AgentTool
from knowledge.prompts.agent_prompt import AGENT_SYSTEM_PROMPT

logger = logging.getLogger(__name__)


class ReActAgent:
    """ReAct 模式的 Agent 引擎。

    多轮 Tool Calling 循环：
      LLM 回复 → 有 tool_calls? → 执行工具 → 把结果加进 messages → 继续
      LLM 回复 → 无 tool_calls? → 返回最终答案
    """

    MAX_TURNS = 5

    def __init__(
        self,
        api_key: str,
        base_url: str,
        model: str,
        tools: List[AgentTool],
        max_turns: int = MAX_TURNS,
        system_prompt: str = AGENT_SYSTEM_PROMPT,
    ):
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._tools = {t.name: t for t in tools}
        self._tool_schemas = [t.tools_json_schema() for t in tools]
        self._max_turns = max_turns
        self._system_prompt = system_prompt
        self._steps: List[Dict[str, Any]] = []

    def get_steps(self) -> List[Dict[str, Any]]:
        """获取 Agent 推理步骤记录。

        Returns:
            每轮步骤的列表，每项包含 turn/thought/actions。
        """
        return self._steps

    def invoke(self, user_query: str, context: str = "", item_names: str = "") -> str:
        self._steps = []
        messages = self._build_initial_messages(user_query, context, item_names)
        final_text = ""

        for turn in range(self._max_turns):
            step = {"turn": turn + 1, "thought": "", "actions": []}

            logger.info(f"{'='*60}")
            logger.info(f"[Agent 第 {turn + 1}/{self._max_turns} 轮]")
            logger.info(f"{'='*60}")

            llm_msg, tool_calls = self._call_llm(messages)

            if llm_msg:
                logger.info(f"💭 Thought (思考): {llm_msg[:300]}{'...' if len(llm_msg) > 300 else ''}")
                step["thought"] = llm_msg

            if tool_calls is None:
                final_text = llm_msg or ""
                logger.info(f"✅ Agent 完成，返回最终答案")
                self._steps.append(step)
                break

            logger.info(f"🔧 本轮计划调用 {len(tool_calls)} 个工具")
            messages.append({"role": "assistant", "content": llm_msg, "tool_calls": tool_calls})

            tool_results = self._execute_tool_calls(tool_calls)
            for tc_id, tool_name, result_text, args in tool_results:
                action_entry = {
                    "tool": tool_name,
                    "args": json.dumps(args, ensure_ascii=False),
                    "observation": result_text[:500] if len(result_text) > 500 else result_text,
                }
                step["actions"].append(action_entry)
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc_id,
                    "name": tool_name,
                    "content": result_text,
                })

            self._steps.append(step)
        else:
            logger.warning("Agent 达到最大轮次限制")
            final_text = messages[-1].get("content", "") if messages else ""

        return final_text or "Agent 暂无法生成答案"

    def stream(
        self, user_query: str, context: str = "", item_names: str = "",
        on_step: Callable[[Dict[str, Any]], None] = None
    ) -> Generator[str, None, None]:
        """流式调用 Agent，yield 增量文本。

        Args:
            user_query: 用户查询。
            context: 知识库上下文。
            item_names: 商品名称。
            on_step: 可选回调，每轮推理完成后调用，传入步骤字典。
        """
        self._steps = []
        messages = self._build_initial_messages(user_query, context, item_names)

        for turn in range(self._max_turns):
            step = {"turn": turn + 1, "thought": "", "actions": []}

            logger.info(f"{'='*60}")
            logger.info(f"[Agent 第 {turn + 1}/{self._max_turns} 轮 - 流式]")
            logger.info(f"{'='*60}")

            llm_msg, tool_calls = self._call_llm_stream(messages)

            if tool_calls is None:
                thought = "".join(llm_msg)
                logger.info(f"💭 Thought (思考): {thought[:300]}{'...' if len(thought) > 300 else ''}")
                logger.info(f"✅ Agent 完成，返回最终答案")
                step["thought"] = thought
                self._steps.append(step)
                if on_step:
                    on_step(step)
                for delta in llm_msg:
                    yield delta
                break

            full_content = "".join(llm_msg)
            logger.info(f"💭 Thought (思考): {full_content[:300]}{'...' if len(full_content) > 300 else ''}")
            step["thought"] = full_content
            logger.info(f"🔧 本轮计划调用 {len(tool_calls)} 个工具")
            messages.append({"role": "assistant", "content": full_content, "tool_calls": tool_calls})

            tool_results = self._execute_tool_calls(tool_calls)
            for tc_id, tool_name, result_text, args in tool_results:
                action_entry = {
                    "tool": tool_name,
                    "args": json.dumps(args, ensure_ascii=False),
                    "observation": result_text[:500] if len(result_text) > 500 else result_text,
                }
                step["actions"].append(action_entry)
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc_id,
                    "name": tool_name,
                    "content": result_text,
                })

            self._steps.append(step)
            if on_step:
                on_step(step)
        else:
            logger.warning("Agent 达到最大轮次限制")

    def _build_initial_messages(
        self, user_query: str, context: str, item_names: str
    ) -> List[Dict[str, Any]]:
        system_msg = self._system_prompt
        if item_names:
            system_msg += f"\n\n当前用户询问的产品: {item_names}"

        return [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_query},
        ]

    def _call_llm(
        self, messages: List[Dict[str, Any]]
    ) -> Tuple[str, Optional[List[Dict[str, Any]]]]:
        """非流式 LLM 调用，返回 (content_text_or_empty, tool_calls_or_None)。"""
        payload = {
            "model": self._model,
            "messages": messages,
            "tools": self._tool_schemas if self._tool_schemas else None,
            "tool_choice": "auto",
            "temperature": 0.1,
        }
        if not self._tool_schemas:
            payload.pop("tools", None)
            payload.pop("tool_choice", None)

        try:
            r = httpx.post(
                f"{self._base_url}/chat/completions",
                headers={
                    "Authorization": f"Bearer {self._api_key}",
                    "Content-Type": "application/json",
                },
                json=payload,
                timeout=120,
            )
            r.raise_for_status()
            data = r.json()
        except Exception as e:
            logger.error(f"Agent LLM call failed: {e}")
            return "", None

        return self._parse_llm_response(data)

    def _call_llm_stream(
        self, messages: List[Dict[str, Any]]
    ) -> Tuple[List[str], Optional[List[Dict[str, Any]]]]:
        """流式 LLM 调用。返回 (增量文本列表, tool_calls_or_None)。"""
        payload = {
            "model": self._model,
            "messages": messages,
            "tools": self._tool_schemas if self._tool_schemas else None,
            "tool_choice": "auto",
            "temperature": 0.1,
            "stream": True,
        }
        if not self._tool_schemas:
            payload.pop("tools", None)
            payload.pop("tool_choice", None)

        content_deltas: List[str] = []
        accumulated_tool_calls: Dict[int, Dict[str, Any]] = {}
        finish_reason = None

        try:
            with httpx.stream(
                "POST",
                f"{self._base_url}/chat/completions",
                headers={
                    "Authorization": f"Bearer {self._api_key}",
                    "Content-Type": "application/json",
                    "Accept": "text/event-stream",
                },
                json=payload,
                timeout=120,
            ) as r:
                for line in r.iter_lines():
                    if not line or not line.startswith("data: "):
                        continue
                    line = line[6:]
                    if line == "[DONE]":
                        break
                    try:
                        chunk = json.loads(line)
                    except JSONDecodeError:
                        continue

                    choices = chunk.get("choices", [])
                    if not choices:
                        continue
                    delta = choices[0].get("delta", {})
                    finish_reason = choices[0].get("finish_reason")

                    if "content" in delta and delta["content"]:
                        content_deltas.append(delta["content"])

                    if "tool_calls" in delta:
                        for tc_delta in delta["tool_calls"]:
                            idx = tc_delta.get("index", 0)
                            if idx not in accumulated_tool_calls:
                                accumulated_tool_calls[idx] = {
                                    "id": tc_delta.get("id", ""),
                                    "type": "function",
                                    "function": {"name": "", "arguments": ""},
                                }
                            tc = accumulated_tool_calls[idx]
                            if tc_delta.get("id"):
                                tc["id"] = tc_delta["id"]
                            if "function" in tc_delta:
                                fn_delta = tc_delta["function"]
                                if "name" in fn_delta:
                                    tc["function"]["name"] += fn_delta["name"] or ""
                                if "arguments" in fn_delta:
                                    tc["function"]["arguments"] += fn_delta["arguments"] or ""
        except Exception as e:
            logger.error(f"Agent LLM stream failed: {e}")
            return content_deltas, None

        if finish_reason == "tool_calls" and accumulated_tool_calls:
            return content_deltas, list(accumulated_tool_calls.values())

        return content_deltas, None

    def _parse_llm_response(
        self, data: Dict[str, Any]
    ) -> Tuple[str, Optional[List[Dict[str, Any]]]]:
        choices = data.get("choices", [])
        if not choices:
            return "", None

        msg = choices[0].get("message", {})
        content = msg.get("content", "") or ""
        tool_calls = msg.get("tool_calls")

        if tool_calls:
            return content, tool_calls
        return content, None

    def _execute_tool_calls(
        self, tool_calls: List[Dict[str, Any]]
    ) -> List[Tuple[str, str, str, Dict[str, Any]]]:
        """执行一组 tool_calls，返回 [(tool_call_id, tool_name, result_text, args_dict)]。"""
        results = []
        for tc in tool_calls:
            tc_id = tc.get("id", "")
            func = tc.get("function", {})
            tool_name = func.get("name", "")
            args_raw = func.get("arguments", "{}")

            tool = self._tools.get(tool_name)
            if not tool:
                results.append((tc_id, tool_name, json.dumps({"error": f"未知工具: {tool_name}"}, ensure_ascii=False), {}))
                continue

            try:
                args = json.loads(args_raw) if isinstance(args_raw, str) else args_raw
            except JSONDecodeError:
                args = {}

            logger.info(f"🔧 Action: 调用工具 {tool_name}({json.dumps(args, ensure_ascii=False)})")
            result_text = tool.safe_run(**args)
            obs_preview = result_text[:500] if len(result_text) > 500 else result_text
            logger.info(f"📋 Observation (结果): {obs_preview}{'...' if len(result_text) > 500 else ''}")
            results.append((tc_id, tool_name, result_text, args))

        return results