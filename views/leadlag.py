"""예측력 — 설명인가 예측인가.

앞 페이지들의 회귀는 전부 같은 봉끼리 맞춘 것이라 설명이지 예측이 아니다.
여기서 기준 코인을 k봉 밀어 회귀해 그 둘을 가른다.

프로파일 차트와 비교표, 수수료 대조는 따로 보면 오독하기 쉬워 한 화면에 둔다.
(R²가 통계적으로 유의해도 수수료보다 작으면 매매로는 못 쓴다.)
"""

from __future__ import annotations

import math
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from core.stats import lead_lag_profile
from core.theme import COIN_COLOR, MUTED, short, style_fig
from core.ui import get_context, page_header

ctx = get_context()
page_header("⏱ 예측력", "설명인가 예측인가", ctx)

label = ctx.market_label
ROUND_TRIP_COST = 0.10  # 업비트 0.05% × 매수·매도

st.markdown(
    f"""
    앞 페이지들의 회귀는 **같은 봉**끼리 맞춘 것이라 *설명*이지 *예측*이 아닙니다.
    여기서는 {label} 수익률을 **k봉 밀어서** 회귀합니다.

    $$r_{{i,t}} = \\alpha + \\beta \\cdot r_{{m,\\,t-k}} + \\varepsilon_t$$

    - **k > 0** — {label}가 **선행**. 값이 크면 *"{label}를 보고 나서 사도 늦지 않다"*
    - **k = 0** — 동시점. 앞 페이지들이 재던 것
    - **k < 0** — 이 코인이 {label}보다 **먼저** 움직임
    """
)

c1, c2 = st.columns([2.4, 1])
with c1:
    picked = st.multiselect("코인 (최대 8개)", ctx.coins, default=ctx.coins[:4],
                            format_func=short, key="ll_coins")
with c2:
    max_lag = st.slider("최대 시차 (±봉)", 1, 15, 5, key="ll_maxlag",
                        help="기준 코인을 앞뒤로 몇 봉까지 밀어볼지")

if not picked:
    st.info("코인을 하나 이상 선택하세요.")
    st.stop()

profiles = {}
for m in picked:
    pair = ctx.rets[[ctx.market, m]].dropna()
    profiles[m] = lead_lag_profile(tuple(pair[m].values), tuple(pair[ctx.market].values), max_lag)


# ======================================================================
# 프로파일
# ======================================================================

fig = go.Figure()
for m, prof in profiles.items():
    fig.add_trace(go.Scatter(
        x=prof["k"], y=prof["r2"], mode="lines+markers", name=short(m),
        line=dict(color=COIN_COLOR[m], width=2),
        marker=dict(size=7, line=dict(width=0)),
        hovertemplate=f"<b>{short(m)}</b><br>k=%{{x}}봉<br>R² = %{{y:.3f}}<extra></extra>",
    ))
fig.add_vline(x=0, line=dict(color=MUTED, width=1))
fig.add_annotation(x=0, y=1.02, yref="paper", text="k=0 (동시점)", showarrow=False,
                   font=dict(color=MUTED, size=10), xanchor="left")
fig.update_xaxes(title_text=f"k — {label}를 몇 봉 밀었나  (오른쪽 = {label} 선행)", dtick=1)
fig.update_yaxes(title_text="R²", tickformat=".0%", rangemode="tozero")
st.plotly_chart(style_fig(fig, height=380), use_container_width=True, key="leadlag")

st.caption(
    "k=0에서 뾰족하게 솟고 양옆으로 급락하면 **동시에 움직일 뿐 예측은 안 된다**는 뜻입니다. "
    "k>0 쪽이 의미 있게 높아야 비로소 따라 사는 전략이 성립합니다."
)


# ======================================================================
# k=0 vs k=+1
# ======================================================================

st.markdown("#### 동시점(k=0) 과 한 봉 뒤(k=+1) 비교")

rows = []
for m, prof in profiles.items():
    p0 = prof[prof["k"] == 0]
    p1 = prof[prof["k"] == 1]
    if p0.empty or p1.empty:
        continue
    r2_0, r2_1 = float(p0["r2"].iloc[0]), float(p1["r2"].iloc[0])
    edge = math.sqrt(max(r2_1, 0.0)) * float(ctx.rets[m].std(ddof=1))
    rows.append({
        "코인": short(m),
        "k=0 R²": r2_0 * 100,
        "k=+1 R²": r2_1 * 100,
        "설명력 감소": ((1 - r2_1 / r2_0) if r2_0 > 0 else np.nan) * 100,
        "k=+1 p값": float(p1["p"].iloc[0]),
        "예측가능 변동폭": edge * 100,
    })

cmp_df = pd.DataFrame(rows)
st.dataframe(
    cmp_df, use_container_width=True, hide_index=True,
    column_config={
        "k=0 R²": st.column_config.NumberColumn(format="%.1f%%", help="같은 봉끼리 맞췄을 때의 설명력"),
        "k=+1 R²": st.column_config.NumberColumn(format="%.2f%%", help="한 봉 밀었을 때 남는 설명력"),
        "설명력 감소": st.column_config.NumberColumn(format="%.1f%%", help="k=0 대비 몇 % 사라졌나"),
        "k=+1 p값": st.column_config.NumberColumn(format="%.3f", help="0.05 미만이면 우연이 아니라고 본다"),
        "예측가능 변동폭": st.column_config.NumberColumn(
            format="%.3f%%",
            help="√R² × 이 코인 변동성. 한 봉당 이론적으로 잡을 수 있는 움직임의 크기"),
    },
)

if not cmp_df.empty:
    best = cmp_df.loc[cmp_df["k=+1 R²"].idxmax()]
    sig = cmp_df[cmp_df["k=+1 p값"] < 0.05]
    edge_max = float(cmp_df["예측가능 변동폭"].max())

    st.markdown(
        f"""
        | 항목 | 값 | 읽는 법 |
        |:---|:---|:---|
        | 설명력 감소 (평균) | **{cmp_df['설명력 감소'].mean():.0f}%** | 한 봉만 밀어도 이만큼 사라집니다 |
        | k=+1 최고 | **{best['코인']} {best['k=+1 R²']:.2f}%** | 남은 설명력이 가장 큰 코인 |
        | p<0.05 인 코인 | **{len(sig)}개 / {len(cmp_df)}개** | 통계적으로 0이 아닌 것 |
        | 최고 예측가능 변동폭 | **{edge_max:.3f}%** | 왕복 수수료 {ROUND_TRIP_COST:.2f}% 와 비교하세요 |
        """
    )

    if edge_max < ROUND_TRIP_COST:
        st.error(
            f"**예측 가능한 움직임({edge_max:.3f}%)이 왕복 수수료({ROUND_TRIP_COST:.2f}%)보다 작습니다.** "
            f"통계적으로 유의하더라도 실제로 매매하면 수수료에 먹힙니다. "
            f"이 구간에서 '{label}를 보고 따라 사는' 전략은 성립하지 않습니다."
        )
    else:
        st.warning(
            f"예측 가능한 움직임({edge_max:.3f}%)이 왕복 수수료({ROUND_TRIP_COST:.2f}%)를 넘습니다. "
            f"다만 슬리피지·체결 지연은 빠져 있고, 아래 다중검정 문제도 확인하세요."
        )

n_tests = (2 * max_lag + 1) * len(profiles)
st.info(
    f"**다중검정 주의.** 시차 {2 * max_lag + 1}개 × 코인 {len(profiles)}개 = "
    f"**{n_tests}번의 검정**을 동시에 하고 있습니다. 유의수준 5%면 아무 관계가 없어도 우연히 "
    f"**{round(n_tests * 0.05)}개 정도**는 'p<0.05'로 나옵니다. "
    f"한두 개가 유의하다고 신호를 찾았다고 보면 안 됩니다."
)
