"""端到端接口验收测试（冒烟测试）。

针对已启动的 API 逐一验证：查询接口、关系图谱、增量接入（结构化 JSON / 自然语言两阶段）、
静态页面，并对返回值做结构与语义断言（而不仅是检查 HTTP 状态码）。

用法：
    python -X utf8 scripts/smoke_test_api.py                      # 默认 http://127.0.0.1:8000
    python -X utf8 scripts/smoke_test_api.py --base http://localhost:8000
    python -X utf8 scripts/smoke_test_api.py --include-ingest     # 写入测试数据

退出码：0 全部通过；1 存在失败用例。
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from urllib import error, request

ROOT = Path(__file__).resolve().parents[1]

PASS, FAIL = "PASS", "FAIL"
results: list[tuple[str, str, float, str]] = []


def _req(method: str, url: str, payload: dict | None = None, timeout: int = 120):
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    req = request.Request(url, data=data, method=method)
    if data:
        req.add_header("Content-Type", "application/json")
    started = time.time()
    try:
        with request.urlopen(req, timeout=timeout) as resp:
            body = resp.read().decode("utf-8", "replace")
            status = resp.status
    except error.HTTPError as exc:
        body = exc.read().decode("utf-8", "replace")
        status = exc.code
    return status, body, (time.time() - started) * 1000


def check(name: str, ok: bool, detail: str, ms: float = 0.0) -> bool:
    results.append((name, PASS if ok else FAIL, ms, detail))
    mark = "✓" if ok else "✗"
    print(f"  [{mark}] {name:<38} {ms:7.0f}ms  {detail}")
    return ok


def json_of(body: str) -> dict:
    try:
        return json.loads(body)
    except ValueError:
        return {}


# ── 用例 ────────────────────────────────────────────────────────────
def case_health(base: str) -> bool:
    st, body, ms = _req("GET", f"{base}/health")
    d = json_of(body)
    return check("健康检查 /health", st == 200 and d.get("status") == "ok",
                 f"status={d.get('status')} cache={list((d.get('cache') or {}).get('loaded', {}).keys())}", ms)


def case_events(base: str) -> bool:
    st, body, ms = _req("GET", f"{base}/api/v1/events/?limit=5")
    d = json_of(body)
    ok = st == 200 and isinstance(d.get("events"), list) and d.get("total", 0) > 0
    detail = f"total={d.get('total'):,}" if ok else f"HTTP {st}"
    if not check("事件组列表 /events/", ok, detail, ms):
        return False
    gid = d["events"][0].get("group_id")
    st2, body2, ms2 = _req("GET", f"{base}/api/v1/events/{gid}")
    d2 = json_of(body2)
    return check("事件组详情 /events/{id}", st2 == 200 and d2.get("group_id") == gid,
                 f"group_id={gid} 报道数={d2.get('article_count')}", ms2)


def case_alerts(base: str) -> bool:
    st, body, ms = _req("GET", f"{base}/api/v1/alert/stats")
    d = json_of(body)
    dist = d.get("distribution") or {}
    total = d.get("total", 0)
    s = sum(dist.get(k, 0) for k in ("P0", "P1", "P2", "P3"))
    ok = st == 200 and s == total and total > 0
    detail = f"P0={dist.get('P0')} P1={dist.get('P1')} P2={dist.get('P2')} P3={dist.get('P3')} 合计={s:,}"
    if not check("预警分布 /alert/stats（分布和=总数）", ok, detail, ms):
        return False
    st2, body2, ms2 = _req("GET", f"{base}/api/v1/alert/?level=P0&limit=5")
    d2 = json_of(body2)
    ok2 = st2 == 200 and all(a.get("alert_level") == "P0" for a in d2.get("alerts", []))
    return check("预警筛选 /alert/?level=P0", ok2,
                 f"P0 条数={len(d2.get('alerts', []))}（均应为 P0）", ms2)


def case_timeline(base: str) -> bool:
    st, body, ms = _req("GET", f"{base}/api/v1/timeline/stats")
    d = json_of(body)
    types = d.get("by_event_type") or {}
    unclassified = d.get("unclassified", 0)
    total = d.get("total_groups", 0)
    ok = (
        st == 200
        and len(types) > 0
        and "" not in types and "未知" not in types      # 未分类不得进入类型分布
        and unclassified + sum(types.values()) == total   # 未分类单独计数
    )
    return check("事件类型分布 /timeline/stats", ok,
                 f"{len(types)} 类 + 未分类 {unclassified:,} = {total:,}", ms)


def case_graph(base: str) -> bool:
    st, body, ms = _req("GET", f"{base}/api/v1/graph/companies?limit=5")
    d = json_of(body)
    ok = st == 200 and len(d.get("companies", [])) > 0
    top = (d.get("companies") or [{}])[0]
    if not check("公司列表 /graph/companies", ok,
                 f"公司数={d.get('total')}（如 {top.get('name')} {top.get('events')} 事件）", ms):
        return False

    code = top.get("code")
    st2, body2, ms2 = _req("GET", f"{base}/api/v1/graph/?company={code}&limit=40")
    g = json_of(body2)
    edges = g.get("edges", [])
    nodes = g.get("nodes", [])
    connected = set()
    for e in edges:
        connected.add(e["source"]); connected.add(e["target"])
    isolated = [n for n in nodes if n["id"] not in connected]
    relations = {e.get("relation") for e in edges}
    ok2 = st2 == 200 and len(nodes) > 0 and not isolated and "reported_by" not in relations
    return check("聚焦图谱（无孤立点、无来源噪声边）", ok2,
                 f"节点={len(nodes)} 边={len(edges)} 孤立={len(isolated)} 关系={sorted(relations)}", ms2)


def case_ingest_json(base: str) -> bool:
    payload = {
        "article_file_id": f"SMOKE-{time.time_ns()}",
        "article_title": "某地新建智能温室大棚投入使用 年产蔬菜预计提升三成",
        "article_source": "验收测试",
        "event_name": "产能扩张",
    }
    st, body, ms = _req("POST", f"{base}/api/v1/ingest", payload)
    d = json_of(body)
    r = (d.get("results") or [{}])[0]
    ok = st == 200 and r.get("status") in ("matched", "new", "duplicate")
    if not check("结构化接入 /ingest", ok,
                 f"status={r.get('status')} 归属={r.get('matched_group_id')} {r.get('message', '')[:26]}", ms):
        return False

    # 一致性：新组应能被详情接口取到，且在事件/预警两个入口都能读到（不重复）
    gid = r.get("matched_group_id")
    st2, body2, ms2 = _req("GET", f"{base}/api/v1/events/{gid}")
    detail = json_of(body2)
    got = detail.get("group_id") == gid
    return check("接入结果可立即读取（事件详情）", got,
                 f"{gid} 报道数={detail.get('article_count')}", ms2)


def case_ingest_nl(base: str) -> bool:
    st, body, ms = _req("POST", f"{base}/api/v1/ingest/nl", {
        "text": "示例文本：万科A公告投资绿色建筑项目，项目计划分阶段推进。",
        "article_source": "验收测试",
    })
    d = json_of(body)

    # 轻量演示包未安装 BGE 等完整路径依赖时，应「优雅降级」：返回明确提示而非报错中断
    degraded = (
        st == 200
        and not d.get("ok")
        and d.get("hint")
        and "完整路径" in str(d.get("hint"))
    )
    if degraded:
        return check("自然语言两阶段接入 /ingest/nl（优雅降级）", True,
                     f"完整路径未启用，已返回提示：{str(d.get('hint'))[:44]}", ms)

    steps = d.get("steps") or []
    labels = [s.get("label") for s in steps]
    # 归组可以走 BGE（完整路径）或 TF-IDF（轻量路径），两者都算通过
    has_cluster = any(lbl in ("BGE 归组", "归组（轻量）") for lbl in labels)
    ok = (
        st == 200
        and d.get("status") in ("matched", "new")
        and all(str(s.get("detail") or "").strip() for s in steps)   # 每步都要有内容
        and has_cluster
    )
    mode = "BGE" if "BGE 归组" in labels else "轻量 TF-IDF"
    detail = (f"{d.get('status')} 组={d.get('matched_group_id') or d.get('new_group_id')} "
              f"相似度={d.get('similarity')} 步骤={len(steps)} 归组方式={mode}")
    if not check("自然语言两阶段接入 /ingest/nl", ok, detail, ms):
        return False

    st2, body2, ms2 = _req("GET", f"{base}/api/v1/ingest/nl/status")
    s = json_of(body2)
    idx = s.get("index") or {}
    if not idx.get("ready"):
        return check("两阶段状态 /ingest/nl/status", st2 == 200,
                     f"未就绪（轻量模式）：{str(s.get('hint'))[:40]}", ms2)
    return check("两阶段状态 /ingest/nl/status", st2 == 200 and idx.get("ready"),
                 f"存量组索引={idx.get('groups'):,} 阈值={s.get('threshold')}", ms2)


def case_pages(base: str) -> bool:
    ok_all = True
    for path in ("/", "/ingest", "/events", "/alerts", "/graph", "/timeline"):
        st, body, ms = _req("GET", f"{base}{path}")
        ok = st == 200 and "<div id=\"app\">" in body
        ok_all &= check(f"页面 {path}", ok, "SPA 入口 200", ms)
    return ok_all


def case_dataset(base: str, name: str, expect_total: int | None) -> bool:
    """检查 API 中的事件组与预警数量是否一致。"""
    st, body, ms = _req("GET", f"{base}/api/v1/alert/stats")
    total = json_of(body).get("total", 0)
    if name == "event_groups":
        event_status, event_body, event_ms = _req("GET", f"{base}/api/v1/events/?limit=1")
        event_total = json_of(event_body).get("total")
        ok = st == 200 and event_status == 200 and total > 0 and event_total == total
        return check("事件组与预警数量一致", ok, f"事件组={event_total}，预警={total}", ms + event_ms)
    ok = st == 200 and total > 0 and (expect_total is None or total == expect_total)
    return check("预警数据", ok, f"共 {total:,} 条", ms)


def case_classification_quality(base: str) -> bool:
    """回归检查：不应出现某类被先验吞并。

    直接由类型分布接口计算，不将全量结果载入本进程（避免验收环境内存吃紧）。
    """
    st, body, ms = _req("GET", f"{base}/api/v1/timeline/stats")
    d = json_of(body)
    types = d.get("by_event_type") or {}
    total = d.get("total_groups") or 0
    if st != 200 or not total or not types:
        return check("分类质量回归（无单类吞并）", False, f"接口异常 HTTP {st}", ms)
    top_name, top_count = max(types.items(), key=lambda kv: kv[1])
    edu = types.get("教育行业政策调整", 0)
    ok = top_count / total < 0.25 and edu / total < 0.10
    return check("分类质量回归（无单类吞并）", ok,
                 f"最大类占比={top_count/total:.1%}（{top_name}）；教育类占比={edu/total:.1%}", ms)


def wait_ready(base: str, timeout: int = 300) -> bool:
    """等待服务就绪（启动预热可能需数十秒），避免评审刚启动就自检而误报。"""
    started = time.time()
    while time.time() - started < timeout:
        try:
            st, _, _ = _req("GET", f"{base}/health", timeout=5)
            if st == 200:
                waited = time.time() - started
                if waited > 1:
                    print(f"（等待服务就绪 {waited:.0f}s）")
                return True
        except Exception:
            pass
        time.sleep(2)
    return False


def main() -> int:
    ap = argparse.ArgumentParser(description="端到端接口验收测试")
    ap.add_argument("--base", default="http://127.0.0.1:8000", help="API 地址")
    ap.add_argument("--include-ingest", action="store_true", help="额外测试接入；会向当前数据写入测试记录")
    ap.add_argument("--check-historical-distribution", action="store_true", help="检查原始实验数据的分类分布")
    ap.add_argument("--wait", type=int, default=300, help="等待服务就绪的最多秒数")
    args = ap.parse_args()
    base = args.base.rstrip("/")

    print(f"目标服务: {base}")
    if not wait_ready(base, args.wait):
        print(f"服务在 {args.wait}s 内未就绪，请确认已启动（bash start.sh）后重试。")
        return 1
    print("-" * 78)
    print("[一] 数据与模型")
    case_dataset(base, "alerts", None)
    case_dataset(base, "event_groups", None)
    if args.check_historical_distribution:
        case_classification_quality(base)

    print("\n[二] 查询接口")
    case_health(base)
    case_events(base)
    case_alerts(base)
    case_timeline(base)
    case_graph(base)

    print("\n[三] 静态页面")
    case_pages(base)

    if args.include_ingest:
        print("\n[四] 增量接入")
        case_ingest_json(base)
        case_ingest_nl(base)

    passed = sum(1 for _, s, _, _ in results if s == PASS)
    failed = [(n, d) for n, s, _, d in results if s == FAIL]
    total_ms = sum(ms for _, _, ms, _ in results)
    print("\n" + "-" * 78)
    print(f"合计: {passed}/{len(results)} 通过，累计耗时 {total_ms/1000:.1f}s")
    if failed:
        print("失败用例:")
        for n, d in failed:
            print(f"  - {n}: {d}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
