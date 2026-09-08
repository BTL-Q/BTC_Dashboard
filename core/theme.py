"""색상 팔레트와 차트 공통 스타일.

팔레트는 dataviz 검증기(validate_palette.js)를 통과한 값이다.
  light: 8슬롯 전부 PASS (contrast WARN → 범례 + 표 뷰로 보완)
  dark : 8슬롯 전부 PASS

색은 "코인(엔티티)"에 고정한다. 정렬 순서나 필터로 절대 바뀌지 않는다.
"""

from __future__ import annotations

import plotly.graph_objects as go
import streamlit as st

from core.config import DEFAULT_COINS

# ----------------------------------------------------------------------
# 테마 판정 — .streamlit/config.toml 에서 base 를 고정해 둔다.
# 고정하지 않으면 UI는 브라우저 설정을 따르는데 팔레트는 light로 잡혀 어긋난다.
# ----------------------------------------------------------------------

DARK = (st.get_option("theme.base") or "light").lower() == "dark"

CAT_LIGHT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100",
             "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
CAT_DARK = ["#3987e5", "#d95926", "#199e70", "#c98500",
            "#d55181", "#008300", "#9085e9", "#e66767"]
CAT = CAT_DARK if DARK else CAT_LIGHT

INK = "#ffffff" if DARK else "#0b0b0b"
INK_2 = "#c3c2b7" if DARK else "#52514e"
MUTED = "#898781"
GRID = "#2c2c2a" if DARK else "#e1e0d9"
AXIS = "#383835" if DARK else "#c3c2b7"
NEUTRAL_MID = "#383835" if DARK else "#f0efec"

FONT = 'system-ui, -apple-system, "Segoe UI", sans-serif'

# 코인 → 고정 색
COIN_COLOR = {m: CAT[i % len(CAT)] for i, m in enumerate(DEFAULT_COINS)}

# 상관행렬용 발산형 스케일: 파랑 ← 중립 회색 → 빨강 (중앙 0 = "관계 없음")
DIVERGING = [
    [0.00, "#0d366b"],
    [0.25, "#3987e5" if DARK else "#2a78d6"],
    [0.50, NEUTRAL_MID],
    [0.75, "#e66767" if DARK else "#e34948"],
    [1.00, "#7f1d1d"],
]


def short(market: str) -> str:
    """KRW-BTC → BTC"""
    return market.split("-")[-1]


def style_fig(fig: go.Figure, height: int = 380, legend: bool = True,
              hovermode: str = "closest") -> go.Figure:
    """모든 차트에 같은 크롬을 입힌다. 얇은 마크, 하이라인 그리드, 투명 배경."""
    fig.update_layout(
        height=height,
        margin=dict(l=10, r=10, t=48 if legend else 16, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family=FONT, color=INK, size=12),
        showlegend=legend,
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0,
            font=dict(color=INK_2, size=11), bgcolor="rgba(0,0,0,0)",
        ),
        hovermode=hovermode,
        hoverlabel=dict(font=dict(family=FONT, size=12)),
        barcornerradius=4,
    )
    axis_kw = dict(
        showgrid=True, gridcolor=GRID, gridwidth=1, zeroline=False,
        linecolor=AXIS, linewidth=1, ticks="",
        tickfont=dict(color=MUTED, size=11),
        title_font=dict(color=MUTED, size=11),
    )
    fig.update_xaxes(**axis_kw)
    fig.update_yaxes(**axis_kw)
    return fig


CARD_CSS = """
<style>
.base-coin-badge {
    border: 1px solid rgba(237, 161, 0, 0.62);
    border-radius: 6px;
    padding: 6px 9px;
    margin-top: 8px;
    background: rgba(237, 161, 0, 0.12);
    text-align: center;
    line-height: 1.05;
}
.base-coin-badge span {
    display: block;
    color: rgba(128,128,128,.95);
    font-size: 11px;
    font-weight: 650;
    margin-bottom: 3px;
}
.base-coin-badge strong {
    display: block;
    color: inherit;
    font-size: 20px;
    font-weight: 800;
}
.coin-card-grid {
    display: grid;
    grid-template-columns: repeat(3, minmax(0, 1fr));
    gap: 12px;
    margin: 10px 0 18px;
}
.coin-card {
    border: 1px solid rgba(128, 128, 128, 0.22);
    border-radius: 8px;
    padding: 14px 16px;
    background: rgba(128, 128, 128, 0.06);
}
.coin-card-label { color: rgba(128,128,128,.95); font-size: 13px; line-height: 1.2; margin-bottom: 8px; }
.coin-card-name  { font-size: 28px; font-weight: 750; line-height: 1.05; margin-bottom: 10px; }
.coin-card-value { font-size: 34px; font-weight: 800; line-height: 1; }
.coin-card-sub   { color: rgba(128,128,128,.95); font-size: 13px; margin-top: 8px; line-height: 1.35; }
@media (max-width: 900px) { .coin-card-grid { grid-template-columns: 1fr; } }

.coin-summary-table { width: 100%; border-collapse: collapse; table-layout: fixed; line-height: 1.3; }
.coin-summary-table th, .coin-summary-table td {
    border-bottom: 1px solid rgba(128,128,128,.22);
    padding: 9px 8px; vertical-align: top;
    word-break: keep-all; overflow-wrap: anywhere;
}
.coin-summary-table th { color: rgba(128,128,128,.95); font-weight: 650; text-align: left; }
.coin-summary-table td:nth-child(1) { width: 16%; color: rgba(128,128,128,.95); }
.coin-summary-table td:nth-child(2) { width: 18%; font-weight: 750; white-space: nowrap; }
.coin-summary-table td:nth-child(3) { width: 66%; }
</style>
"""
