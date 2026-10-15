"""Shared UI helpers: CSS theme, cards, chart styling."""
from __future__ import annotations

import html

import plotly.graph_objects as go
import streamlit as st

ACCENT, ACCENT2, MUTED = "#22D3EE", "#8B5CF6", "#94A3B8"

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap');
html, body, [class*="css"], .stApp { font-family: 'Inter', system-ui, sans-serif; }
.stApp { background: radial-gradient(1200px 600px at 85% -10%, rgba(139,92,246,.14), transparent 60%),
                     radial-gradient(900px 500px at -10% 0%, rgba(34,211,238,.10), transparent 55%), #0A0F1C; }
.block-container { padding-top: 1.6rem; padding-bottom: 3rem; max-width: 1280px; }
h1, h2, h3 { letter-spacing: -0.02em; }
section[data-testid="stSidebar"] { background: #0D1424; border-right: 1px solid rgba(148,163,184,.12); }
.brand { display:flex; align-items:center; gap:.7rem; margin:.2rem 0 1rem; }
.brand .logo { width:38px; height:38px; border-radius:11px; display:flex; align-items:center; justify-content:center;
  background: linear-gradient(135deg, #22D3EE, #8B5CF6); font-weight:800; color:#06111f; font-size:1.05rem; }
.brand .t { font-weight:800; font-size:1.15rem; line-height:1.1; } .brand .s { color:#94A3B8; font-size:.72rem; }
.hero { padding: 1.7rem 1.8rem; border-radius: 20px; margin-bottom: 1.2rem; position: relative; overflow: hidden;
  background: linear-gradient(135deg, rgba(34,211,238,.14), rgba(139,92,246,.16) 60%, rgba(244,114,182,.10));
  border: 1px solid rgba(148,163,184,.18); }
.hero h1 { margin:0 0 .35rem; font-size: 2.05rem; font-weight: 800; }
.hero p { margin:0; color:#CBD5E1; font-size: 1.02rem; max-width: 880px; line-height:1.55; }
.chips { margin-top: .9rem; display:flex; flex-wrap:wrap; gap:.45rem; }
.chip { font-size:.74rem; padding:.22rem .65rem; border-radius:999px; border:1px solid rgba(148,163,184,.28);
  color:#CBD5E1; background: rgba(15,23,42,.55); }
.kpi { background: rgba(17,24,39,.78); border:1px solid rgba(148,163,184,.14); border-radius:16px; padding:.95rem 1.05rem;
  height:100%; border-top: 3px solid var(--c, #22D3EE); }
.kpi .l { color:#94A3B8; font-size:.74rem; text-transform:uppercase; letter-spacing:.07em; font-weight:600; }
.kpi .v { font-size:1.75rem; font-weight:800; margin:.15rem 0 .05rem; line-height:1.1; }
.kpi .s { color:#94A3B8; font-size:.76rem; }
.card { background: rgba(17,24,39,.78); border:1px solid rgba(148,163,184,.14); border-radius:16px; padding:1.05rem 1.2rem; height:100%; }
.card h4 { margin:.1rem 0 .45rem; font-size:1rem; } .card p, .card li { color:#CBD5E1; font-size:.9rem; line-height:1.55; }
.card ul { padding-left:1.1rem; margin:.2rem 0; }
.sec { margin: 1.6rem 0 .3rem; font-size:1.25rem; font-weight:700; } .cap { color:#94A3B8; font-size:.9rem; margin-bottom:.7rem; }
.callout { border-left: 4px solid var(--c, #22D3EE); background: rgba(34,211,238,.07); padding:.75rem 1rem; border-radius:0 12px 12px 0;
  color:#E2E8F0; font-size:.92rem; margin:.7rem 0; line-height:1.55; }
.badge { display:inline-block; padding:.18rem .7rem; border-radius:999px; font-weight:700; font-size:.8rem; color:#0A0F1C; background:var(--c); }
.flow { display:flex; align-items:stretch; gap:.4rem; flex-wrap:wrap; margin:.4rem 0 .6rem; }
.step { flex:1 1 150px; background: rgba(17,24,39,.85); border:1px solid rgba(148,163,184,.16); border-radius:14px; padding:.8rem .9rem; }
.step b { color:#22D3EE; font-size:.72rem; letter-spacing:.08em; } .step div { font-weight:600; font-size:.92rem; margin-top:.15rem; }
.step span { color:#94A3B8; font-size:.78rem; } .arrow { align-self:center; color:#475569; font-size:1.3rem; }
.mono { font-family: 'JetBrains Mono', monospace; }
div[data-testid="stTabs"] button { font-weight:600; }
.stButton > button { border-radius:12px; font-weight:600; }
#MainMenu, footer { visibility: hidden; }
</style>
"""


def inject_css() -> None:
    st.markdown(CSS, unsafe_allow_html=True)


def hero(title: str, subtitle: str, chips: list[str] | None = None) -> None:
    ch = "".join(f'<span class="chip">{html.escape(c)}</span>' for c in (chips or []))
    st.markdown(f'<div class="hero"><h1>{title}</h1><p>{subtitle}</p><div class="chips">{ch}</div></div>', unsafe_allow_html=True)


def section(title: str, caption: str = "") -> None:
    st.markdown(f'<div class="sec">{title}</div>' + (f'<div class="cap">{caption}</div>' if caption else ""), unsafe_allow_html=True)


def kpis(items: list[tuple]) -> None:
    """items: (label, value, sub, colour)"""
    cols = st.columns(len(items))
    for col, (label, value, sub, color) in zip(cols, items):
        col.markdown(f'<div class="kpi" style="--c:{color}"><div class="l">{label}</div><div class="v">{value}</div>'
                     f'<div class="s">{sub}</div></div>', unsafe_allow_html=True)


def card(title: str, body_html: str) -> str:
    return f'<div class="card"><h4>{title}</h4>{body_html}</div>'


def callout(text: str, color: str = ACCENT) -> None:
    st.markdown(f'<div class="callout" style="--c:{color}">{text}</div>', unsafe_allow_html=True)


def badge(text: str, color: str) -> str:
    return f'<span class="badge" style="--c:{color}">{html.escape(text)}</span>'


def flow(steps: list[tuple[str, str, str]]) -> None:
    parts = []
    for i, (tag, name, sub) in enumerate(steps):
        parts.append(f'<div class="step"><b>{tag}</b><div>{name}</div><span>{sub}</span></div>')
        if i < len(steps) - 1:
            parts.append('<div class="arrow">&rsaquo;</div>')
    st.markdown(f'<div class="flow">{"".join(parts)}</div>', unsafe_allow_html=True)


def style(fig: go.Figure, height: int = 380, legend: bool = True) -> go.Figure:
    fig.update_layout(
        template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", height=height,
        font=dict(family="Inter, sans-serif", color="#CBD5E1", size=12), margin=dict(l=8, r=8, t=44, b=8),
        title=dict(font=dict(size=15, color="#F1F5F9"), x=0.0), showlegend=legend,
        legend=dict(bgcolor="rgba(0,0,0,0)", orientation="h", y=-0.18))
    fig.update_xaxes(gridcolor="rgba(148,163,184,.12)", zerolinecolor="rgba(148,163,184,.18)")
    fig.update_yaxes(gridcolor="rgba(148,163,184,.12)", zerolinecolor="rgba(148,163,184,.18)")
    return fig


def show(fig: go.Figure, **kw) -> None:
    try:
        st.plotly_chart(fig, width="stretch", **kw)
    except TypeError:  # older Streamlit
        st.plotly_chart(fig, use_container_width=True, **kw)


def table(df, **kw) -> None:
    try:
        st.dataframe(df, width="stretch", hide_index=True, **kw)
    except TypeError:
        st.dataframe(df, use_container_width=True, hide_index=True, **kw)
