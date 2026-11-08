"""
FastAPI 应用入口
"""
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from src.config import settings

# 加载项目根目录 .env（若存在）：便于配置 LLM API Key 等可选增强项。
# 已写入 .gitignore，不会进入版本库。
try:
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parents[2] / ".env")
except Exception:  # python-dotenv 缺失时静默跳过
    pass

# 前端构建产物；存在时由 FastAPI 直接托管，实现单端口启动。
FRONTEND_DIST = Path(__file__).resolve().parents[2] / "frontend" / "dist"

# index.html 不允许缓存（否则浏览器会一直引用旧的分片，改版后看不到新界面）；
# 带内容哈希的 assets 由 StaticFiles 常规缓存。
_NO_CACHE = {"Cache-Control": "no-cache, no-store, must-revalidate", "Pragma": "no-cache"}

# 缺前端构建产物时的提示（避免只看到 FastAPI 默认的 {"detail": "Not Found"}）
MISSING_DIST_HINT = (
    "前端未构建（frontend/dist 不存在）：Web 界面不可用，仅提供 API。"
    "若只想直接运行查看效果，请使用演示包 项目源码；"
    "若使用源代码包，请在 frontend 目录执行 npm install && npm run build 后重启。"
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """启动时预热结果缓存。

    全量结果文件较大（轻量版仍有数百 MB），若在首个请求时才解析，
    响应会达到 10s+。启动阶段同步预热可保证后续请求稳定在毫秒级。
    可通过 EVENTRADAR_WARMUP=0 关闭。
    """
    if os.environ.get("EVENTRADAR_WARMUP", "1") != "0":
        from src.api.data_store import load_results

        for name in ("event_groups", "alerts"):
            try:
                rows = load_results(name)
                print(f"[warmup] {name}: {len(rows):,} 条已缓存")
            except Exception as exc:  # 预热失败不应阻止服务启动
                print(f"[warmup] {name} 加载失败: {exc}")

        # 预热在单个后台线程中串行执行，不阻塞服务启动。
        # 串行而非并行：两项预热均属 CPU/内存密集（TF-IDF 建索引 + BGE/分类器加载），
        # 并发时内存峰值叠加、可能引发进程异常退出；串行后峰值明显下降。
        import threading

        def _init_nl() -> None:
            if os.environ.get("EVENTRADAR_NL_WARMUP", "1") == "0":
                return
            try:
                from src.services.group_vector_index import REPS_PATH

                if not REPS_PATH.exists():
                    print("[warmup] 两阶段接入：无存量组索引，跳过预热（轻量模式）")
                    return
                from src.services.nl_pipeline import get_nl_pipeline

                service = get_nl_pipeline()
                service.status()            # 加载存量组向量索引
                _ = service.embedder.model  # 加载 BGE（本机缓存，不联网）
                _ = service.classifier      # 加载事件分类器
                print("[warmup] 两阶段接入就绪：BGE + 已有事件组索引 + 事件分类器")
            except Exception as exc:  # 预热失败不影响服务
                print(f"[warmup] 两阶段接入预热跳过: {type(exc).__name__}: {exc}")

        def _warmup_all() -> None:
            try:
                from src.services.ingest_service import get_ingest_service

                service = get_ingest_service()
                service.build_index()
                replayed = service.replay_log()
                print(f"[warmup] 增量索引就绪: {service.status()['groups_indexed']:,} 组，重放 {replayed} 条接入记录")
            except Exception as exc:
                print(f"[warmup] 增量索引构建失败: {type(exc).__name__}: {exc}")
            _init_nl()

        threading.Thread(target=_warmup_all, daemon=True, name="warmup").start()
    yield


app = FastAPI(
    title="Event Radar API",
    description="上市公司事件识别、聚类、可信度评估与预警系统",
    version="0.1.0",
    lifespan=lifespan,
)

# CORS 配置（允许前端跨域）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health_check():
    from src.api.data_store import cache_info

    return {"status": "ok", "service": "event-radar", "cache": cache_info()}


@app.get("/")
def root():
    """有前端构建产物时返回 Web 入口，否则返回 API 信息。"""
    index = FRONTEND_DIST / "index.html"
    if index.exists():
        return FileResponse(index, headers=_NO_CACHE)
    return {
        "name": "Event Radar API",
        "version": "0.1.0",
        "docs": "/docs",
        "notice": MISSING_DIST_HINT,
    }


# 路由注册：四类接口均读取端到端流水线生成的真实结果文件。
from src.api.routers import events, timeline, graph, alert, ingest

app.include_router(events.router, prefix="/api/v1/events", tags=["events"])
app.include_router(timeline.router, prefix="/api/v1/timeline", tags=["timeline"])
app.include_router(graph.router, prefix="/api/v1/graph", tags=["graph"])
app.include_router(alert.router, prefix="/api/v1/alert", tags=["alert"])
app.include_router(ingest.router, prefix="/api/v1/ingest", tags=["ingest"])


# ── 前端静态资源托管 ──
# 注册在所有 API 路由之后：具体 API 路径优先匹配，其余交给 SPA（history 模式回退到 index.html）。
if FRONTEND_DIST.exists():
    _assets = FRONTEND_DIST / "assets"
    if _assets.exists():
        app.mount("/assets", StaticFiles(directory=_assets), name="assets")

    @app.get("/{full_path:path}", include_in_schema=False)
    def spa_fallback(full_path: str):
        if full_path.startswith(("api/", "docs", "openapi.json", "redoc")):
            return {"detail": "Not Found"}
        candidate = (FRONTEND_DIST / full_path).resolve()
        if candidate.is_file() and FRONTEND_DIST.resolve() in candidate.parents:
            return FileResponse(candidate)
        return FileResponse(FRONTEND_DIST / "index.html", headers=_NO_CACHE)

else:
    # 无前端构建产物：给出明确提示，而不是默认 404
    @app.get("/{full_path:path}", include_in_schema=False)
    def spa_missing(full_path: str):
        if full_path.startswith(("api/", "docs", "openapi.json", "redoc")):
            return {"detail": "Not Found"}
        return JSONResponse(status_code=503, content={"detail": MISSING_DIST_HINT, "api_docs": "/docs"})
