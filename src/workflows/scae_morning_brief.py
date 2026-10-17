"""SCAE Morning Signal Report — daily strategic signal brief published to Notion.

Workflow steps:
1. Fetch the last 48 hours of Iran-related news from Google News RSS (English and Persian feeds).
2. Synthesize a concise Persian morning signal brief with the Mistral API.
3. Publish the report as a daily page under the strategic knowledge parent page in Notion.
"""

from __future__ import annotations

import datetime as dt
import json
from email.utils import parsedate_to_datetime
from typing import Any
from urllib.parse import quote_plus
from xml.etree import ElementTree as ET
from zoneinfo import ZoneInfo

import httpx
import mistralai.workflows as workflows
from mistralai.workflows import Depends
from mistralai.workflows.client import get_mistral_client
from mistralai.workflows.plugins.mistralai.connectors import (
    ToolCallClient,
    connector,
    uses_connectors,
)

QATAR_TZ = ZoneInfo("Asia/Qatar")
MODEL = "mistral-medium-latest"
NOTION_PARENT_PAGE_ID = "3c51bfb5-0d1d-8180-a835-e929813e481d"
MAX_ITEMS = 30
MIN_RECENT = 8
FETCH_TIMEOUT = 25.0

FEED_QUERIES = [
    ("Iran Strait of Hormuz", "en"),
    ("Iran war negotiations", "en"),
    ("ایران هرمز", "fa"),
]

FA_DIGITS = str.maketrans("0123456789", "۰۱۲۳۴۵۶۷۸۹")
GMONTHS = [
    "ژانویه", "فوریه", "مارس", "آوریل", "مه", "ژوئن",
    "ژوئیه", "اوت", "سپتامبر", "اکتبر", "نوامبر", "دسامبر",
]
JMONTHS = [
    "فروردین", "اردیبهشت", "خرداد", "تیر", "مرداد", "شهریور",
    "مهر", "آبان", "آذر", "دی", "بهمن", "اسفند",
]
WEEKDAYS = ["دوشنبه", "سه‌شنبه", "چهارشنبه", "پنج‌شنبه", "جمعه", "شنبه", "یکشنبه"]


def to_fa(value: Any) -> str:
    """Convert digits in a value to Persian digits."""
    return str(value).translate(FA_DIGITS)


def gregorian_to_jalali(gy: int, gm: int, gd: int) -> tuple[int, int, int]:
    """Convert a Gregorian date to the Jalali (Solar Hijri) calendar."""
    g_d_m = [0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334]
    gy2 = gy + 1 if gm > 2 else gy
    days = (
        355666 + (365 * gy)
        + ((gy2 + 3) // 4)
        - ((gy2 + 99) // 100)
        + ((gy2 + 399) // 400)
        + gd + g_d_m[gm - 1]
    )
    jy = -1595 + (33 * (days // 12053))
    days = days % 12053
    jy += 4 * (days // 1461)
    days = days % 1461
    if days > 365:
        jy += (days - 1) // 365
        days = (days - 1) % 365
    if days < 186:
        jm = 1 + days // 31
        jd = 1 + days % 31
    else:
        jm = 7 + (days - 186) // 30
        jd = 1 + (days - 186) % 30
    return jy, jm, jd


def build_title(now: dt.datetime) -> str:
    return "SCAE | Morning Signal Report — " + to_fa(now.day) + " " + GMONTHS[now.month - 1] + " " + to_fa(now.year)


def build_date_line(now: dt.datetime) -> str:
    jy, jm, jd = gregorian_to_jalali(now.year, now.month, now.day)
    greg = WEEKDAYS[now.weekday()] + " " + to_fa(now.day) + " " + GMONTHS[now.month - 1] + " " + to_fa(now.year)
    jalali = to_fa(jd) + " " + JMONTHS[jm - 1] + " " + to_fa(jy)
    return greg + " | " + jalali + " | دوحه (UTC+3)"


def _feed_url(query: str, lang: str) -> str:
    encoded = quote_plus(query + " when:2d")
    if lang == "fa":
        return "https://news.google.com/rss/search?q=" + encoded + "&hl=fa&gl=IR&ceid=IR:fa"
    return "https://news.google.com/rss/search?q=" + encoded + "&hl=en-US&gl=US&ceid=US:en"


@workflows.activity(name="fetch-news", start_to_close_timeout=dt.timedelta(seconds=90))
async def fetch_news(now_iso: str) -> str:
    """Fetch and deduplicate recent Iran-related news; return a text digest."""
    now = dt.datetime.fromisoformat(now_iso)
    cutoff = now - dt.timedelta(hours=48)
    seen: set[str] = set()
    items: list[dict[str, Any]] = []

    async with httpx.AsyncClient(timeout=FETCH_TIMEOUT, follow_redirects=True) as client:
        for query, lang in FEED_QUERIES:
            url = _feed_url(query, lang)
            try:
                response = await client.get(url)
                response.raise_for_status()
                root = ET.fromstring(response.text)
            except Exception:
                continue
            for item in root.findall("./channel/item"):
                title = (item.findtext("title") or "").strip()
                if not title:
                    continue
                key = title.lower()
                if key in seen:
                    continue
                seen.add(key)
                source_element = item.find("source")
                source = (
                    (source_element.text or "").strip()
                    if source_element is not None and source_element.text
                    else ""
                )
                published = None
                raw_date = (item.findtext("pubDate") or "").strip()
                if raw_date:
                    try:
                        published = parsedate_to_datetime(raw_date)
                    except Exception:
                        published = None
                items.append(
                    {
                        "title": title,
                        "source": source or "Google News",
                        "published": published,
                        "sort_key": published.timestamp() if published else now.timestamp(),
                    }
                )

    if not items:
        raise RuntimeError("هیچ فید خبری در دسترس نبود؛ جمع‌آوری اخبار ناموفق بود.")

    recent = [it for it in items if it["published"] is None or it["published"] >= cutoff]
    selected = recent if len(recent) >= MIN_RECENT else items
    selected.sort(key=lambda it: it["sort_key"], reverse=True)
    selected = selected[:MAX_ITEMS]

    lines = []
    for it in selected:
        when = (
            it["published"].strftime("%Y-%m-%d %H:%M") + " UTC"
            if it["published"] is not None
            else "بدون زمان"
        )
        lines.append("- " + it["title"] + " | منبع: " + it["source"] + " | زمان: " + when)
    return "\n".join(lines)


SYSTEM_PROMPT = (
    "تو تحلیلگر ارشد سامانه راهبردی SCAE هستی و گزارش سیگنال صبحگاهی روزانه تولید می‌کنی. "
    "زبان گزارش فارسی رسمی، صریح و بدون غلط املایی است. "
    "ساختار دقیق متن گزارش در قالب مارک‌داون: "
    "یک پاراگراف موجز درباره تصویر کلان میدان (حداکثر سه خط)؛ "
    "سپس تیتر پنج سیگنال صبحگاهی با دقیقا پنج آیتم، که هر آیتم یک جمله خبر و یک جمله تحلیل کوتاه دارد و نام منبع را ذکر می‌کند؛ "
    "سپس تیتر شاخص تشدید با یک عدد از صفر تا ده و یک جمله توجیه؛ "
    "سپس تیتر هشدار ۲۴ ساعته با دقیقا یک شاخص مشخص برای پایش در ۲۴ ساعت آینده و دلیل آن؛ "
    "و در پایان تیتر هشتگ‌ها با سه تا شش هشتگ فارسی مرتبط. "
    "از احساساتی‌نویسی و ادعای قطعی بدون شواهد پرهیز کن و در صورت نبود قطعیت از زبان احتمالی استفاده کن. "
    "خروجی فقط باید JSON معتبر با دو کلید باشد: escal_index به صورت عدد صحیح از صفر تا ده، و content به صورت متن کامل مارک‌داون گزارش."
)


@workflows.activity(name="synthesize-brief", start_to_close_timeout=dt.timedelta(seconds=240))
async def synthesize_brief(news_digest: str, title: str, date_line: str) -> dict[str, Any]:
    """Synthesize the morning signal brief from the news digest."""
    client = get_mistral_client()
    user_message = (
        "عنوان گزارش: " + title + "\n"
        "خط تاریخ: " + date_line + "\n\n"
        "اخبار ۴۸ ساعت گذشته:\n" + news_digest
    )
    response = await client.chat.complete_async(
        model=MODEL,
        temperature=0.3,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
    )
    raw = (response.choices[0].message.content or "").strip()
    fence = chr(96) * 3
    if raw.startswith(fence):
        raw = raw[len(fence):]
        if raw[:4].lower() == "json":
            raw = raw[4:]
        raw = raw.strip()
    if raw.endswith(fence):
        raw = raw[: -len(fence)].strip()
    data = json.loads(raw)
    escal_index = int(data.get("escal_index", 5))
    content = str(data.get("content", "")).strip()
    if not content:
        raise RuntimeError("محتوای گزارش تولیدشده خالی بود.")
    return {"escal_index": escal_index, "content": content}


notion_connector = connector("notion")


@workflows.activity(name="publish-notion-page", start_to_close_timeout=dt.timedelta(seconds=120))
async def publish_notion_page(
    title: str,
    content: str,
    notion: ToolCallClient = Depends(notion_connector),
) -> dict[str, Any]:
    """Publish the report as a page under the strategic knowledge parent page in Notion."""
    result = await notion.call_tool(
        tool_name="notion-create-pages",
        arguments={
            "pages": [
                {
                    "properties": {"title": title},
                    "icon": "\U0001F9ED",
                    "content": content,
                }
            ],
            "parent": {"type": "page_id", "page_id": NOTION_PARENT_PAGE_ID},
            "allow_async": False,
            "_confirmationReason": "ثبت گزارش سیگنال صبحگاهی روزانه SCAE در Notion",
        },
    )
    return {"published": True, "result": result}


@workflows.workflow.define(
    name="scae-morning-brief",
    workflow_display_name="SCAE Morning Signal Report",
    workflow_description="گزارش سیگنال صبحگاهی روزانه SCAE — جمع‌آوری اخبار، سنتز تحلیلی و ثبت روزانه در Notion",
)
@uses_connectors(notion_connector)
class ScaeMorningBriefWorkflow:
    @workflows.workflow.entrypoint
    async def run(self) -> dict[str, Any]:
        now = dt.datetime.now(QATAR_TZ)
        title = build_title(now)
        date_line = build_date_line(now)
        news_digest = await fetch_news(now.isoformat())
        brief = await synthesize_brief(news_digest=news_digest, title=title, date_line=date_line)
        content = "**" + date_line + "**\n\n" + brief["content"]
        await publish_notion_page(title=title, content=content)
        return {"title": title, "escal_index": brief["escal_index"]}
