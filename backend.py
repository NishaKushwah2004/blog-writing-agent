from __future__ import annotations

import operator
import os
import re
from datetime import date, timedelta
from pathlib import Path
from typing import TypedDict, List, Optional, Literal, Annotated

from pydantic import BaseModel, Field

from langgraph.graph import StateGraph, START, END
from langgraph.types import Send

from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage
from dotenv import load_dotenv

load_dotenv()

# ============================================================
# Blog Writer (Router → (Research?) → Orchestrator → Workers → ReducerWithImages)
# Patches image capability using your 3-node reducer flow:
#   merge_content -> decide_images -> generate_and_place_images
# ============================================================


# -----------------------------
# 1) Schemas
# -----------------------------
class Task(BaseModel):
    id: int
    title: str
    goal: str = Field(..., description="One sentence describing what the reader should do/understand.")
    bullets: List[str] = Field(..., min_length=3, max_length=6)
    target_words: int = Field(..., description="Target words (120–550).")

    tags: List[str] = Field(default_factory=list)
    requires_research: bool = False
    requires_citations: bool = False
    requires_code: bool = False


class Plan(BaseModel):
    blog_title: str
    audience: str
    tone: str
    blog_kind: Literal["explainer", "tutorial", "news_roundup", "comparison", "system_design"] = "explainer"
    constraints: List[str] = Field(default_factory=list)
    tasks: List[Task]


class EvidenceItem(BaseModel):
    title: str
    url: str
    published_at: Optional[str] = None  # ISO "YYYY-MM-DD" preferred
    snippet: Optional[str] = None
    source: Optional[str] = None


class RouterDecision(BaseModel):
    needs_research: bool
    mode: Literal["closed_book", "hybrid", "open_book"]
    reason: str
    queries: List[str] = Field(default_factory=list)
    max_results_per_query: int = Field(5)


class EvidencePack(BaseModel):
    evidence: List[EvidenceItem] = Field(default_factory=list)


# ---- Image planning schema (ported from your image flow) ----
class ImageSpec(BaseModel):
    placeholder: str = Field(..., description="e.g. [[IMAGE_1]]")
    filename: str = Field(..., description="Save under images/, e.g. qkv_flow.png")
    alt: str
    caption: str
    prompt: str = Field(..., description="Prompt to send to the image model.")
    size: Literal["1024x1024", "1024x1536", "1536x1024"] = "1024x1024"
    quality: Literal["low", "medium", "high"] = "medium"


class GlobalImagePlan(BaseModel):
    md_with_placeholders: str
    images: List[ImageSpec] = Field(default_factory=list)

class State(TypedDict):
    topic: str

    # routing / research
    mode: str
    needs_research: bool
    queries: List[str]
    evidence: List[EvidenceItem]
    plan: Optional[Plan]

    # recency
    as_of: str
    recency_days: int

    # workers
    sections: Annotated[List[tuple[int, str]], operator.add]  # (task_id, section_md)

    # reducer/image
    merged_md: str
    md_with_placeholders: str
    image_specs: List[dict]

    final: str


# -----------------------------
# 2) LLM
# -----------------------------

router_llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0,
)

research_llm = ChatGroq(
    model="qwen/qwen3-32b",
    temperature=0,
)

planner_llm = ChatGroq(
    model="qwen/qwen3-32b",
    temperature=0.2,
    max_tokens=1000,
)

writer_llm = ChatGroq(
    model="qwen/qwen3-32b",
    temperature=0.3,
    max_tokens=800,
)

image_llm = ChatGroq(
    model="qwen/qwen3-32b",
    temperature=0.2,
    max_tokens=500,
)
# -----------------------------
# 3) Router
# -----------------------------
ROUTER_SYSTEM = """You are a routing module for a technical blog planner.

Decide whether web research is needed BEFORE planning.

Modes:
- closed_book (needs_research=false): evergreen concepts.
- hybrid (needs_research=true): evergreen + needs up-to-date examples/tools/models.
- open_book (needs_research=true): volatile weekly/news/"latest"/pricing/policy.

If needs_research=true:
- Output 3–10 high-signal, scoped queries.
- For open_book weekly roundup, include queries reflecting last 7 days.
"""

def router_node(state: State) -> dict:
    decider = router_llm.with_structured_output(RouterDecision)
    decision = decider.invoke(
        [
            SystemMessage(content=ROUTER_SYSTEM),
            HumanMessage(content=f"Topic: {state['topic']}\nAs-of date: {state['as_of']}"),
        ]
    )

    if decision.mode == "open_book":
        recency_days = 7
    elif decision.mode == "hybrid":
        recency_days = 45
    else:
        recency_days = 3650

    return {
        "needs_research": decision.needs_research,
        "mode": decision.mode,
        "queries": decision.queries,
        "recency_days": recency_days,
    }

def route_next(state: State) -> str:
    return "research" if state["needs_research"] else "orchestrator"

# -----------------------------
# 4) Research (Tavily)
# -----------------------------
def _tavily_search(query: str, max_results: int = 3) -> List[dict]:
    if not os.getenv("TAVILY_API_KEY"):
        return []
    try:
        from langchain_tavily import TavilySearch  # type: ignore
        tool = TavilySearch(max_results=max_results)
        results = tool.invoke({"query": query})
        out: List[dict] = []
        for r in results or []:
            out.append(
                {
                    "title": r.get("title") or "",
                    "url": r.get("url") or "",
                    "snippet": r.get("content") or r.get("snippet") or "",
                    "published_at": r.get("published_date") or r.get("published_at"),
                    "source": r.get("source"),
                }
            )
        return out
    except Exception:
        return []

def _iso_to_date(s: Optional[str]) -> Optional[date]:
    if not s:
        return None
    try:
        return date.fromisoformat(s[:10])
    except Exception:
        return None

RESEARCH_SYSTEM = """
You are an expert research analyst.

Given search results, return ONLY the most relevant EvidenceItem objects.

Rules:
- Keep only authoritative sources.
- Remove duplicates.
- Ignore low-quality websites.
- Keep snippets under 150 characters.
- Normalize dates to YYYY-MM-DD if available.
- Return at most 8 EvidenceItems.
"""


def research_node(state: State) -> dict:
    queries = (state.get("queries") or [])[:5]

    raw: List[dict] = []

    for q in queries:
        raw.extend(_tavily_search(q, max_results=2))

    if not raw:
        return {"evidence": []}

    # -----------------------------------
    # Compress search results
    # -----------------------------------
    compact_results = []
    seen_urls = set()

    for r in raw:
        url = r.get("url")

        if not url or url in seen_urls:
            continue

        seen_urls.add(url)

        compact_results.append(
            {
                "title": r.get("title", ""),
                "url": url,
                "snippet": (r.get("snippet", "")[:180]).strip(),
                "published_at": r.get("published_at"),
            }
        )

    # Prevent sending huge prompts
    compact_results = compact_results[:10]

    extractor = research_llm.with_structured_output(EvidencePack)

    pack = extractor.invoke(
        [
            SystemMessage(content=RESEARCH_SYSTEM),
            HumanMessage(
                content=f"""
As-of date: {state['as_of']}

Recency days: {state['recency_days']}

Search Results:

{compact_results}
"""
            ),
        ]
    )

    # -----------------------------------
    # Deduplicate evidence
    # -----------------------------------
    dedup = {}

    for e in pack.evidence:
        if e.url:
            dedup[e.url] = e

    evidence = list(dedup.values())

    # -----------------------------------
    # Filter recent evidence for open-book mode
    # -----------------------------------
    if state.get("mode") == "open_book":
        as_of = date.fromisoformat(state["as_of"])
        cutoff = as_of - timedelta(days=int(state["recency_days"]))

        evidence = [
            e
            for e in evidence
            if (d := _iso_to_date(e.published_at)) and d >= cutoff
        ]

    return {
        "evidence": evidence[:8]
    }

# -----------------------------
# 5) Orchestrator (Plan)
# -----------------------------
ORCH_SYSTEM = """
You are a senior technical writer and developer advocate.

Your job is to create the BEST possible outline for a technical blog.

Requirements:
- Produce 5-8 sections.
- Each section must have:
    - title
    - goal
    - 3-5 bullets
    - target_words
- Use evidence only when necessary.
- Do not invent facts.
- Keep the outline concise and logical.

Grounding:
- closed_book → evergreen knowledge only.
- hybrid → use evidence where appropriate.
- open_book → news roundup only.

Output MUST follow the Plan schema.
"""


def orchestrator_node(state: State) -> dict:

    planner = planner_llm.with_structured_output(Plan)

    mode = state.get("mode", "closed_book")
    evidence = state.get("evidence", [])

    forced_kind = "news_roundup" if mode == "open_book" else None

    # -----------------------------------
    # Compress evidence before sending
    # -----------------------------------

    planner_evidence = []

    for e in evidence[:5]:
        planner_evidence.append(
            {
                "title": e.title,
                "snippet": (e.snippet or "")[:120],
                "url": e.url,
            }
        )

    plan = planner.invoke(
        [
            SystemMessage(content=ORCH_SYSTEM),
            HumanMessage(
                content=f"""
Topic:
{state['topic']}

Mode:
{mode}

As-of:
{state['as_of']}

Recency:
{state['recency_days']} days

{f'Force blog_kind=news_roundup' if forced_kind else ''}

Available Evidence:

{planner_evidence}
"""
            ),
        ]
    )

    if forced_kind:
        plan.blog_kind = "news_roundup"

    return {
        "plan": plan
    }


# -----------------------------
# 6) Fanout
# -----------------------------
def fanout(state: State):
    assert state["plan"] is not None

    MAX_SECTIONS = 5

    return [
        Send(
            "worker",
            {
                "task": task.model_dump(),
                "topic": state["topic"],
                "mode": state["mode"],
                "as_of": state["as_of"],
                "recency_days": state["recency_days"],
                "plan": state["plan"].model_dump(),
                "evidence": [
                    e.model_dump()
                    for e in state.get("evidence", [])[:3]
                ],
            },
        )
        for task in state["plan"].tasks[:MAX_SECTIONS]
    ]

# -----------------------------
# 7) Worker
# -----------------------------
WORKER_SYSTEM = """
You are an expert technical blog writer.

Write exactly ONE markdown section.

Requirements:
- Begin with: ## <Section Title>
- Cover every bullet in order.
- Be concise and technically accurate.
- Stay close to the target word count.
- Use bullet lists where appropriate.
- Use tables only if they improve clarity.
- Include code only when requires_code=True.
- Cite only the supplied evidence URLs if requires_citations=True.
- Do not repeat content from other sections.
"""

def worker_node(payload: dict) -> dict:
    task = Task(**payload["task"])
    plan = Plan(**payload["plan"])

    evidence = [
        EvidenceItem(**e)
        for e in payload.get("evidence", [])
    ]

    bullets = "\n".join(f"- {b}" for b in task.bullets)

    evidence_text = "\n".join(
        f"- {e.title}\n  {e.url}"
        for e in evidence[:3]
    )

    prompt = f"""
Title: {plan.blog_title}

Topic: {payload['topic']}

Section: {task.title}

Goal:
{task.goal}

Target words:
{min(task.target_words, 300)}

Bullets:
{bullets}

Needs citations:
{task.requires_citations}

Needs code:
{task.requires_code}

Evidence:
{evidence_text}
"""

    section = writer_llm.invoke(
        [
            SystemMessage(content=WORKER_SYSTEM),
            HumanMessage(content=prompt),
        ]
    ).content.strip()

    return {
        "sections": [(task.id, section)]
    }


# ============================================================
# 8) ReducerWithImages (subgraph)
#    merge_content -> decide_images -> generate_and_place_images
# ============================================================
def merge_content(state: State) -> dict:
    plan = state["plan"]
    if plan is None:
        raise ValueError("merge_content called without plan.")
    ordered_sections = [md for _, md in sorted(state["sections"], key=lambda x: x[0])]
    body = "\n\n".join(ordered_sections).strip()
    merged_md = f"# {plan.blog_title}\n\n{body}\n"
    return {"merged_md": merged_md}


DECIDE_IMAGES_SYSTEM = """
You are an expert technical editor.

Your task is to decide whether a technical blog would benefit from diagrams.

Rules:
- Maximum 3 images.
- Only recommend diagrams that improve understanding.
- Prefer:
    • Architecture diagrams
    • Flowcharts
    • Comparison tables
    • Pipelines
    • System designs
    • Concept illustrations
- Never recommend decorative images.
- Insert placeholders:
    [[IMAGE_1]]
    [[IMAGE_2]]
    [[IMAGE_3]]
- If no images are useful, return the original markdown unchanged.
- Return ONLY GlobalImagePlan.
"""

def decide_images(state: State) -> dict:

    planner = image_llm.with_structured_output(GlobalImagePlan)

    plan = state["plan"]
    assert plan is not None

    merged_md = state["merged_md"]

    # ---------------------------------------
    # Reduce prompt size
    # ---------------------------------------

    preview = merged_md[:5000]

    image_plan = planner.invoke(
        [
            SystemMessage(content=DECIDE_IMAGES_SYSTEM),
            HumanMessage(
                content=f"""
Blog kind:
{plan.blog_kind}

Topic:
{state['topic']}

Below is the blog.

Decide whether technical diagrams should be inserted.

{preview}
"""
            ),
        ]
    )

    return {
        "md_with_placeholders": image_plan.md_with_placeholders,
        "image_specs": [
            img.model_dump()
            for img in image_plan.images
        ],
    }


def _gemini_generate_image_bytes(prompt: str) -> bytes:
    """
    Returns raw image bytes generated by Gemini.
    Requires: pip install google-genai
    Env var: GOOGLE_API_KEY
    """
    from google import genai
    from google.genai import types

    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("GOOGLE_API_KEY is not set.")

    client = genai.Client(api_key=api_key)

    resp = client.models.generate_content(
        model="gemini-2.5-flash-image",
        contents=prompt,
        config=types.GenerateContentConfig(
            response_modalities=["IMAGE"],
            safety_settings=[
                types.SafetySetting(
                    category="HARM_CATEGORY_DANGEROUS_CONTENT",
                    threshold="BLOCK_ONLY_HIGH",
                )
            ],
        ),
    )

    # Depending on SDK version, parts may hang off resp.candidates[0].content.parts
    parts = getattr(resp, "parts", None)
    if not parts and getattr(resp, "candidates", None):
        try:
            parts = resp.candidates[0].content.parts
        except Exception:
            parts = None

    if not parts:
        raise RuntimeError("No image content returned (safety/quota/SDK change).")

    for part in parts:
        inline = getattr(part, "inline_data", None)
        if inline and getattr(inline, "data", None):
            return inline.data

    raise RuntimeError("No inline image bytes found in response.")


def _safe_slug(title: str) -> str:
    s = title.strip().lower()
    s = re.sub(r"[^a-z0-9 _-]+", "", s)
    s = re.sub(r"\s+", "_", s).strip("_")
    return s or "blog"


def generate_and_place_images(state: State) -> dict:
    plan = state["plan"]
    assert plan is not None

    md = state.get("md_with_placeholders") or state["merged_md"]
    image_specs = state.get("image_specs", []) or []

    # If no images requested, just write merged markdown
    if not image_specs:
        filename = f"{_safe_slug(plan.blog_title)}.md"
        Path(filename).write_text(md, encoding="utf-8")
        return {"final": md}

    images_dir = Path("images")
    images_dir.mkdir(exist_ok=True)

    for spec in image_specs:
        placeholder = spec["placeholder"]
        filename = spec["filename"]
        out_path = images_dir / filename

        # generate only if needed
        if not out_path.exists():
            try:
                img_bytes = _gemini_generate_image_bytes(spec["prompt"])
                out_path.write_bytes(img_bytes)
            except Exception as e:
                # graceful fallback: keep doc usable
                prompt_block = (
                    f"> **[IMAGE GENERATION FAILED]** {spec.get('caption','')}\n>\n"
                    f"> **Alt:** {spec.get('alt','')}\n>\n"
                    f"> **Prompt:** {spec.get('prompt','')}\n>\n"
                    f"> **Error:** {e}\n"
                )
                md = md.replace(placeholder, prompt_block)
                continue

        img_md = f"![{spec['alt']}](images/{filename})\n*{spec['caption']}*"
        md = md.replace(placeholder, img_md)

    filename = f"{_safe_slug(plan.blog_title)}.md"
    Path(filename).write_text(md, encoding="utf-8")
    return {"final": md}

# build reducer subgraph
reducer_graph = StateGraph(State)
reducer_graph.add_node("merge_content", merge_content)
reducer_graph.add_node("decide_images", decide_images)
reducer_graph.add_node("generate_and_place_images", generate_and_place_images)
reducer_graph.add_edge(START, "merge_content")
reducer_graph.add_edge("merge_content", "decide_images")
reducer_graph.add_edge("decide_images", "generate_and_place_images")
reducer_graph.add_edge("generate_and_place_images", END)
reducer_subgraph = reducer_graph.compile()

# -----------------------------
# 9) Build main graph
# -----------------------------
g = StateGraph(State)
g.add_node("router", router_node)
g.add_node("research", research_node)
g.add_node("orchestrator", orchestrator_node)
g.add_node("worker", worker_node)
g.add_node("reducer", reducer_subgraph)

g.add_edge(START, "router")
g.add_conditional_edges("router", route_next, {"research": "research", "orchestrator": "orchestrator"})
g.add_edge("research", "orchestrator")

g.add_conditional_edges("orchestrator", fanout, ["worker"])
g.add_edge("worker", "reducer")
g.add_edge("reducer", END)

app = g.compile()
app
