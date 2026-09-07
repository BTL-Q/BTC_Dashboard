"""안정성 — 관계가 시간에 따라 얼마나 흔들리는가.

롤링 β와 롤링 R²는 반드시 같이 봐야 한다.
β만 보면 "크게 반응한다"까지만 알고, R²를 같이 봐야
"그런데 그 반응이 일관적인가"를 알 수 있다.
"""

from __future__ import annotations

import plotly.graph_objects as go
import streamlit as st

from core.stats import rolling_beta_r2
from core.theme import COIN_COLOR, MUTED, short, style_fig
from core.ui import GLOSSARY, get_context, page_header

ctx = get_context()
page_header("📈 안정성", "관계가 시간에 따라 얼마나 흔들리나", ctx)

label = ctx.market_label

c1, c2 = st.columns([2.4, 1])
with c1:
    picked = st.multiselect(
        "코인 (최대 8개)", ctx.coins, default=ctx.coins[:4],
        format_func=short, key="roll_coins",
    )
with c2:
    default_win = 90 if ctx.timeframe == "1d" else 720
    max_win = max(60, min(len(ctx.rets) // 2, 1500))
    window = st.slider("롤링 윈도우 (봉)", 30, max_win, min(default_win, max_win),
                       step=10, key="roll_window", help=GLOSSARY["rolling"])

if not picked:
    st.info("코인을 하나 이상 선택하세요.")
    st.stop()

beta_lines, r2_lines = {}, {}
for m in picked:
    pair = ctx.rets[[ctx.market, m]].dropna()
    b, rr = rolling_beta_r2(pair[m], pair[ctx.market], window)
    beta_lines[m] = b.dropna()
    r2_lines[m] = rr.dropna()


def rolling_chart(series_map, title, yfmt, ref=None, ref_label=None):
    fig = go.Figure()
    for m, s in series_map.items():
        fig.add_trace(go.Scatter(
            x=s.index, y=s.values, mode="lines", name=short(m),
            line=dict(color=COIN_COLOR[m], width=2),
            hovertemplate=f"<b>{short(m)}</b><br>%{{x|%Y-%m-%d}}<br>{title} %{{y:.2f}}<extra></extra>",
        ))
    if ref is not None:
        fig.add_hline(y=ref, line=dict(color=MUTED, width=1))
        fig.add_annotation(x=1.0, xref="paper", y=ref, text=ref_label, showarrow=False,
                           font=dict(color=MUTED, size=10), xanchor="right", yanchor="bottom")
    # 계열 4개 이하면 끝점 직접 라벨 — 색만으로 식별하지 않게
    if len(series_map) <= 4:
        for m, s in series_map.items():
            if len(s):
                fig.add_annotation(x=s.index[-1], y=float(s.iloc[-1]), text=f" {short(m)}",
                                   showarrow=False, xanchor="left",
                                   font=dict(color=COIN_COLOR[m], size=11))
    fig.update_yaxes(title_text=title, tickformat=yfmt)
    fig.update_xaxes(title_text="")
    return fig


fig_b = rolling_chart(beta_lines, "β", ".2f", ref=1.0, ref_label="β=1")
st.plotly_chart(style_fig(fig_b, height=340, hovermode="x unified"),
                use_container_width=True, key="roll_beta")

st.markdown(
    f"""
    | 롤링 beta에서 볼 것 | 쉬운 해석 |
    |---|---|
    | 1보다 위 | {label}보다 더 크게 움직이는 구간 |
    | 1 근처 | {label}와 비슷한 폭으로 움직이는 구간 |
    | 1보다 아래 | {label}보다 덜 움직이는 구간 |
    | 선이 자주 출렁임 | beta가 안정적이지 않음. 분석 구간에 따라 결론이 바뀔 수 있음 |
    """
)
st.caption(f"현재 설정은 최근 {window}봉만 잘라서 beta를 계속 다시 계산합니다.")

fig_r = rolling_chart(r2_lines, "R²", ".0%")
fig_r.update_yaxes(range=[0, 1])
st.plotly_chart(style_fig(fig_r, height=340, hovermode="x unified"),
                use_container_width=True, key="roll_r2")

st.markdown(
    f"""
    | 롤링 R²에서 볼 것 | 쉬운 해석 |
    |---|---|
    | 위로 올라감 | {label} 영향력이 커지는 구간 |
    | 아래로 내려감 | 코인 자체 움직임이 커지는 구간 |
    | 여러 코인이 동시에 높아짐 | 시장이 한 방향으로 묶이는 구간. 분산 효과가 줄어듦 |
    | beta는 높은데 R²는 낮음 | 크게 반응하긴 하지만, {label}를 꾸준히 따라간다고 보긴 어려움 |
    """
)
st.caption("롤링 회귀는 과거 전체 평균이 아니라, 지금 관계가 예전과 같은지 보는 용도입니다.")


# ======================================================================
# 흔들림 정도를 숫자로
# ======================================================================

st.markdown("#### 베타가 얼마나 흔들렸나")

rows = []
for m, s in beta_lines.items():
    if len(s) < 2:
        continue
    full = ctx.results[m]["beta"]
    rows.append({
        "코인": short(m),
        "전체구간 β": full,
        "롤링 최소": float(s.min()),
        "롤링 최대": float(s.max()),
        "변동폭": float(s.max() - s.min()),
        "표준편차": float(s.std(ddof=1)),
    })

if rows:
    import pandas as pd

    st.dataframe(
        pd.DataFrame(rows), use_container_width=True, hide_index=True,
        column_config={
            "전체구간 β": st.column_config.NumberColumn(
                format="%.2f", help="이 구간 전체를 한 번에 회귀했을 때의 β"),
            "롤링 최소": st.column_config.NumberColumn(format="%.2f"),
            "롤링 최대": st.column_config.NumberColumn(format="%.2f"),
            "변동폭": st.column_config.NumberColumn(
                format="%.2f", help="최대 - 최소. 클수록 '이 코인의 베타'라는 고정값이 없다는 뜻"),
            "표준편차": st.column_config.NumberColumn(format="%.2f"),
        },
    )
    worst = max(rows, key=lambda x: x["변동폭"])
    st.warning(
        f"**{worst['코인']}** 의 β가 {worst['롤링 최소']:.2f} ~ {worst['롤링 최대']:.2f} 사이를 오갔습니다 "
        f"(변동폭 {worst['변동폭']:.2f}). 전체구간 회귀값 {worst['전체구간 β']:.2f} 하나만 보고 "
        f"사이징을 정하면 실제와 크게 어긋날 수 있습니다."
    )
