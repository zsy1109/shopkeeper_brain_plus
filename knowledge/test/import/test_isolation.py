"""
知识库与网络内容隔离 - 端到端验证测试

前置条件：
    1. FastAPI 服务器已启动
    2. MongoDB 中已有 '万用表的使用.pdf' 导入记录

用法：
    cd D:/pycharm_projects/shopkeeper_brain_plus
    python -m knowledge.test.import.test_isolation
"""

import json
import os
import sys
import urllib.request
import urllib.error
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

BASE_URL = "http://127.0.0.1:8001"
SESSION_PREFIX = f"test_iso_{datetime.now().strftime('%Y%m%d%H%M%S')}"


def _query(question: str, enable_agent: bool = True) -> dict:
    """调用 /query 接口（非流式），返回完整响应。"""
    body = json.dumps({
        "query": question,
        "session_id": f"{SESSION_PREFIX}_{hash(question) & 0xFFFF:04x}",
        "is_stream": False,
        "enable_agent": enable_agent,
    }).encode("utf-8")

    req = urllib.request.Request(
        f"{BASE_URL}/query",
        data=body,
        method="POST",
        headers={
            "Content-Type": "application/json; charset=utf-8",
            "Accept": "application/json; charset=utf-8",
        },
    )
    try:
        resp = urllib.request.urlopen(req, timeout=120)
        return resp.status, json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        return e.code, {"detail": e.read().decode("utf-8", errors="replace")}


def _check_answer(question: str, answer: str, expect_kb: bool, expect_web: bool):
    """检查答案是否符合预期。"""
    has_kb_tag = "【知识库原文】" in answer
    has_web_tag = "【网络补充信息】" in answer
    has_conflict = "以知识库" in answer and "冲突" in answer

    kb_ok = has_kb_tag == expect_kb
    web_ok = has_web_tag == expect_web

    checks = []
    if expect_kb:
        checks.append(("含【知识库原文】", has_kb_tag))
    else:
        checks.append(("不含【知识库原文】", not has_kb_tag))
    if expect_web:
        checks.append(("含【网络补充信息】", has_web_tag))
    else:
        checks.append(("不含【网络补充信息】", not has_web_tag))

    detail = {
        "question": question,
        "answer_preview": answer[:300],
        "has_kb_tag": has_kb_tag,
        "has_web_tag": has_web_tag,
        "has_conflict": has_conflict,
        "checks": dict(checks),
    }
    return all(v for _, v in checks), detail


def print_separator(title: str):
    print(f"\n{'=' * 65}")
    print(f"  {title}")
    print(f"{'=' * 65}")


def main():
    print_separator("知识库与网络内容隔离 - 验证测试")
    print(f"  时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # ── 测试用例 1：知识库 PDF 文档内有答案 ──
    print_separator("测试用例1: PDF文档内有答案 → 应只引用知识库，不调用网络")
    print("  问题: 万用表如何测量直流电压？")

    code1, resp1 = _query("万用表如何测量直流电压？", enable_agent=True)
    answer1 = resp1.get("answer", "")

    print(f"  HTTP 状态码: {code1}")
    print(f"  答案长度: {len(answer1)} 字符")

    ok1, detail1 = _check_answer(
        "万用表如何测量直流电压？",
        answer1,
        expect_kb=True,
        expect_web=False,
    )

    # 补充检查：答案不应为空或拒绝
    if len(answer1) < 50:
        ok1 = False
        detail1["checks"]["答案长度≥50"] = False
        print(f"  ✗ 答案过短: {answer1[:200]}")
    else:
        detail1["checks"]["答案长度≥50"] = True

    # 检查 Agent steps 中是否调用了 web 工具
    agent_steps1 = resp1.get("agent_steps", [])
    web_called = any(
        "search_web" in str(step) or "get_realtime_price" in str(step)
        for step in (agent_steps1 or [])
    )
    detail1["agent_web_called"] = web_called
    if not web_called:
        detail1["checks"]["未调用网络工具"] = True
        print(f"  ✓ 未调用 search_web 工具")
    else:
        detail1["checks"]["未调用网络工具"] = False
        ok1 = False
        print(f"  ✗ 调用了网络工具")

    print(f"  答案预览: {answer1[:250]}...")
    print(f"  检查结果: {'✓ 通过' if ok1 else '✗ 失败'}")

    # ── 测试用例 2：文档没有价格信息，应允许联网 ──
    print_separator("测试用例2: 文档无价格信息 → 应标注【网络补充信息】")
    print("  问题: 数字万用表的最新市场价格和购买渠道是什么？")

    code2, resp2 = _query("数字万用表的最新市场价格和购买渠道是什么？", enable_agent=True)
    answer2 = resp2.get("answer", "")

    print(f"  HTTP 状态码: {code2}")
    print(f"  答案长度: {len(answer2)} 字符")

    ok2, detail2 = _check_answer(
        "数字万用表的最新市场价格和购买渠道是什么？",
        answer2,
        expect_kb=True,
        expect_web=True,
    )

    print(f"  答案预览: {answer2[:250]}...")
    print(f"  检查结果: {'✓ 通过' if ok2 else '✗ 失败'}")

    # ── 测试用例 3：知识库内容与网络信息冲突 → 以知识库为准 ──
    print_separator("测试用例3: 文档与网络内容冲突 → 以知识库 PDF 为准")
    print("  问题: 数字万用表的精度参数与规格是什么？如果网络有不同说法请以知识库为准")

    code3, resp3 = _query(
        "数字万用表的精度参数和规格是什么？如果网络上有不同说法请以知识库为准",
        enable_agent=True,
    )
    answer3 = resp3.get("answer", "")

    print(f"  HTTP 状态码: {code3}")
    print(f"  答案长度: {len(answer3)} 字符")

    ok3, detail3 = _check_answer(
        "万用表规格参数",
        answer3,
        expect_kb=True,
        expect_web=False,
    )

    # 对于测试3，关键验证点：
    #   - 必须有【知识库原文】
    #   - 不应有编造的规格参数
    #   - 如果调用了 web，应有冲突声明
    has_conflict_declaration = any(kw in answer3 for kw in [
        "以知识库", "知识库为准", "冲突", "仅供参考"
    ])
    detail3["has_conflict_declaration"] = has_conflict_declaration

    print(f"  答案预览: {answer3[:250]}...")
    print(f"  检查结果: {'✓ 通过' if ok3 else '✗ 失败'}")

    # ── 汇总报告 ──
    results = [
        ("测试用例1: 知识库有答案 → 仅引用KB", ok1),
        ("测试用例2: 知识库无价格 → 允许联网", ok2),
        ("测试用例3: 冲突时以KB为准", ok3),
    ]

    print_separator("测 试 报 告")
    for name, ok in results:
        status = "✓ 通过" if ok else "✗ 失败"
        print(f"  {status}  {name}")

    passed = sum(1 for _, ok in results if ok)
    print(f"\n  结果: {passed}/{len(results)} 通过")

    # 详细输出
    print(f"\n{'=' * 65}")
    print("  详细答案")
    print(f"{'=' * 65}")

    for label, answer in [
        ("测试1 - 仅KB", answer1),
        ("测试2 - KB+联网", answer2),
        ("测试3 - 冲突优先KB", answer3),
    ]:
        truncated = answer[:500] + ("..." if len(answer) > 500 else "")
        print(f"\n--- {label} ---")
        print(truncated)

    return 0 if passed == 3 else 1


if __name__ == "__main__":
    sys.exit(main())