"""End-to-end test for Agent mode toggle feature"""
import requests
import time
import json
import sys

BASE_QUERY = "http://localhost:8001"

def wait_for_completion(task_id, timeout=120):
    """Poll /status/{task_id} until completed"""
    start = time.time()
    while time.time() - start < timeout:
        try:
            r = requests.get(f"{BASE_QUERY}/status/{task_id}", timeout=10)
            data = r.json()
            status = data.get("status", "")
            if status == "completed":
                return data
            elif status == "failed":
                return data
            print(f"  Poll: status={status}, running={data.get('running_list', [])}")
            time.sleep(3)
        except Exception as e:
            print(f"  Poll error: {e}")
            time.sleep(3)
    return {"status": "timeout"}


def test_normal_rag():
    """Test 1: Normal RAG mode without Agent"""
    print("\n" + "=" * 60)
    print("TEST 1: 普通RAG模式 (enable_agent=False)")
    print("提问：数字万用表有哪些警告")
    print("=" * 60)

    payload = {
        "query": "数字万用表有哪些警告",
        "is_stream": False,
        "enable_agent": False
    }

    r = requests.post(f"{BASE_QUERY}/query", json=payload, timeout=120)
    print(f"HTTP {r.status_code}")

    if r.status_code != 200:
        print(f"FAIL: HTTP {r.status_code}: {r.text}")
        return False

    resp = r.json()
    print(f"session_id: {resp.get('session_id', 'N/A')}")
    print(f"task_id: {resp.get('task_id', 'N/A')}")

    task_id = resp.get("task_id", "")
    if task_id:
        print("Waiting for completion...")
        data = wait_for_completion(task_id)
        status = data.get("status", "")
        if status == "completed":
            answer = data.get("answer", "")
            print(f"PASS: Answer length={len(answer)}")
            print(f"Answer:\n{answer[:800]}")
            return True
        else:
            print(f"FAIL: status={status}, error={data.get('error', '')}")
            return False
    else:
        answer = resp.get("answer", "")
        print(f"PASS: Direct answer length={len(answer)}")
        print(f"Answer:\n{answer[:800]}")
        return True


def test_agent_mode():
    """Test 2: Agent mode with ReAct"""
    print("\n" + "=" * 60)
    print("TEST 2: Agent模式 (enable_agent=True)")
    print("提问：数字万用表有哪些安全警告")
    print("=" * 60)

    payload = {
        "query": "数字万用表有哪些安全警告",
        "is_stream": False,
        "enable_agent": True
    }

    r = requests.post(f"{BASE_QUERY}/query", json=payload, timeout=120)
    print(f"HTTP {r.status_code}")

    if r.status_code != 200:
        print(f"FAIL: HTTP {r.status_code}: {r.text}")
        return False

    resp = r.json()
    print(f"session_id: {resp.get('session_id', 'N/A')}")
    print(f"task_id: {resp.get('task_id', 'N/A')}")

    task_id = resp.get("task_id", "")
    if task_id:
        print("Waiting for completion...")
        data = wait_for_completion(task_id)
        status = data.get("status", "")
        if status == "completed":
            answer = data.get("answer", "")
            print(f"PASS: Answer length={len(answer)}")
            print(f"Answer:\n{answer[:800]}")
            return True
        else:
            print(f"FAIL: status={status}, error={data.get('error', '')}")
            return False
    else:
        answer = resp.get("answer", "")
        print(f"PASS: Direct answer length={len(answer)}")
        print(f"Answer:\n{answer[:800]}")
        return True


if __name__ == "__main__":
    print("\n" + "#" * 60)
    print("# Agent模式切换 端到端测试")
    print("#" * 60)

    results = []

    # Test 1: Normal RAG
    try:
        ok = test_normal_rag()
        results.append(("Test 1: 普通RAG (enable_agent=False)", "PASS" if ok else "FAIL"))
    except Exception as e:
        results.append(("Test 1: 普通RAG (enable_agent=False)", f"ERROR: {e}"))

    # Test 2: Agent mode
    try:
        ok = test_agent_mode()
        results.append(("Test 2: Agent模式 (enable_agent=True)", "PASS" if ok else "FAIL"))
    except Exception as e:
        results.append(("Test 2: Agent模式 (enable_agent=True)", f"ERROR: {e}"))

    # Summary
    print("\n" + "=" * 60)
    print("TEST REPORT")
    print("=" * 60)
    for name, status in results:
        print(f"  [{status}] {name}")

    all_pass = all("PASS" in s for _, s in results)
    print(f"\n整体结果: {'全部通过' if all_pass else '存在失败'}")
    sys.exit(0 if all_pass else 1)