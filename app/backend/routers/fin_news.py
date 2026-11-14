"""
FinNews router — 財金新聞教學分析
====================================
Flow: Tavily 搜尋 → Claude 教學分析 → 回傳 JSON
前端負責存入 Supabase。
"""
import os
import logging
from datetime import date

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/fin-news", tags=["FinNews"])

TAVILY_API_KEY    = os.getenv("TAVILY_API_KEY", "")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
LLM_MODEL         = os.getenv("LLM_MODEL", "claude-sonnet-4-5")

SEARCH_QUERIES = [
    "台股 重大消息 今日 影響股市",
    "Fed 聯準會 利率 美股 最新",
    "台灣 政治 政策 經濟 影響股市",
    "半導體 AI 科技股 今日新聞",
    "全球 通膨 景氣 市場風險 最新",
]

SYSTEM_PROMPT = """你是一位資深財經老師，專門用清楚易懂的方式幫助學生理解複雜的財經新聞。
你的風格：
- 親切但專業，像在課堂上講解
- 善於舉具體例子說明抽象概念
- 重視學習過程，不只給結論
- 對台股、美股和總體經濟都有深入了解
- 重視宏觀趨勢與個股的具體連結"""

USER_PROMPT_TEMPLATE = """以下是今天 ({date}) 收集到的財金相關新聞（涵蓋台股、美股、政治、總經）：

{news_content}

請用**繁體中文**完成以下分析，格式要清楚整齊：

## 📌 今日重點事件摘要
列出 3-5 個最重要的事件，每個用一句話說明核心內容。

## 🔍 深度解析
針對每個重點事件分析（每個事件獨立一個小節）：

**📰 [事件標題]**
- **發生了什麼**：（清楚說明事件本身）
- **背後原因**：（為什麼會發生，歷史背景）
- **可能影響**：（對哪些產業/股票，短期vs長期影響）
- **學習重點**：（這個事件讓我們了解什麼財經知識）
- **來源**：請附上相關新聞連結

## 🎯 台股今日關注方向
根據以上分析，今日台股哪些族群/個股值得特別留意？給出具體方向和理由。

語氣：像老師在課堂解說，舉具體例子，避免過度專業術語。
⚠️ 本分析由 AI 生成，僅供學習參考，不構成投資建議。"""


class ArticleItem(BaseModel):
    title:  str
    url:    str
    source: str


class AnalyzeResponse(BaseModel):
    date:     str
    articles: list[ArticleItem]
    analysis: str
    model:    str


@router.post("/analyze", response_model=AnalyzeResponse)
async def analyze_news():
    """Tavily 搜尋 + Claude 教學分析"""
    if not TAVILY_API_KEY:
        raise HTTPException(status_code=500, detail="TAVILY_API_KEY not configured on server")
    if not ANTHROPIC_API_KEY:
        raise HTTPException(status_code=500, detail="ANTHROPIC_API_KEY not configured on server")

    # ── 1. Tavily Search ─────────────────────────────────────────
    try:
        from tavily import TavilyClient
        tc = TavilyClient(api_key=TAVILY_API_KEY)
    except ImportError:
        raise HTTPException(status_code=500, detail="tavily-python not installed")

    all_articles: list[dict] = []
    for query in SEARCH_QUERIES:
        try:
            result = tc.search(
                query=query,
                search_depth="basic",
                max_results=4,
                include_answer=False,
            )
            for r in result.get("results", []):
                all_articles.append({
                    "title":   r.get("title", ""),
                    "url":     r.get("url", ""),
                    "content": r.get("content", "")[:500],
                    "source":  r.get("source", r.get("url", "")[:40]),
                    "query":   query,
                })
        except Exception as e:
            logger.warning(f"Tavily search failed for '{query}': {e}")

    # Deduplicate by URL
    seen: set[str] = set()
    unique: list[dict] = []
    for a in all_articles:
        if a["url"] not in seen and a["title"]:
            seen.add(a["url"])
            unique.append(a)

    if not unique:
        raise HTTPException(status_code=503, detail="No articles retrieved from Tavily")

    logger.info(f"Tavily fetched {len(unique)} unique articles")

    # ── 2. Build prompt content ──────────────────────────────────
    news_lines = []
    for i, a in enumerate(unique[:20], 1):
        news_lines.append(
            f"[{i}] **{a['title']}**\n"
            f"來源: {a['source']} | URL: {a['url']}\n"
            f"摘要: {a['content']}\n"
        )
    news_content = "\n".join(news_lines)

    # ── 3. Claude Analysis ────────────────────────────────────────
    try:
        import anthropic
        ac = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
        response = ac.messages.create(
            model=LLM_MODEL,
            max_tokens=3000,
            system=SYSTEM_PROMPT,
            messages=[{
                "role":    "user",
                "content": USER_PROMPT_TEMPLATE.format(
                    date=date.today().isoformat(),
                    news_content=news_content,
                ),
            }],
        )
        analysis = response.content[0].text
        logger.info(f"Claude analysis complete ({len(analysis)} chars)")
    except Exception as e:
        logger.error(f"Claude error: {e}")
        raise HTTPException(status_code=500, detail=f"Claude API error: {e}")

    return AnalyzeResponse(
        date=date.today().isoformat(),
        articles=[
            ArticleItem(title=a["title"], url=a["url"], source=a["source"])
            for a in unique[:20]
        ],
        analysis=analysis,
        model=LLM_MODEL,
    )


@router.get("/health")
async def fin_news_health():
    return {
        "tavily":    bool(TAVILY_API_KEY),
        "anthropic": bool(ANTHROPIC_API_KEY),
        "model":     LLM_MODEL,
    }
