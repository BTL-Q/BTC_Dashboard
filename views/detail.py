"""코인 상세 — 한 코인을 골라 지표·산점도·해석을 함께 본다.

지표 숫자와 산점도, 그 해석은 따로 보면 의미가 없어서 한 화면에 묶는다.
"""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from core.theme import CAT, COIN_COLOR, MUTED, short, style_fig
from core.ui import GLOSSARY, get_context, page_header

ctx = get_context()
page_header("🔍 코인 상세", "한 코인을 골라 자세히", ctx)

label = ctx.market_label

focus = st.selectbox("코인", ctx.coins, index=0, format_func=short, key="focus_coin")
r = ctx.results[focus]


# ======================================================================
# 지표
# ======================================================================

m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("β (민감도)", f"{r['beta']:.2f}",
          f"±{(r['beta_hi'] - r['beta']):.2f} (95% CI)", delta_color="off",
          help=GLOSSARY["beta"])
m2.metric("R² (설명력)", f"{r['r2']:.1%}", help=GLOSSARY["r2"])
m3.metric("ρ (상관)", f"{r['rho']:.2f}", help=GLOSSARY["rho"])
m4.metric("α (연율)", f"{r['alpha_ann']:+.1%}",
          "유의" if r["p_alpha"] < 0.05 else "무의미", delta_color="off",
          help=GLOSSARY["alpha"])
m5.metric("고유 변동성", f"{r['resid_vol']:.1%}",
          f"총 {r['total_vol']:.0%} 중", delta_color="off",
          help=GLOSSARY["resid_vol"])


# ======================================================================
# 산점도 — 지표가 어디서 나왔는지 눈으로
# ======================================================================

pair = ctx.rets[[ctx.market, focus]].dropna()
xs, ys = pair[ctx.market].values, pair[focus].values
line_x = np.array([xs.min(), xs.max()])
line_y = r["alpha_bar"] + r["beta"] * line_x

fig = go.Figure()
fig.add_trace(go.Scatter(
    x=xs, y=ys, mode="markers", name=f"{short(focus)} 봉별 수익률",
    marker=dict(color=COIN_COLOR[focus], size=7, opacity=0.40, line=dict(width=0)),
    hovertemplate=f"{label} %{{x:.2%}}<br>" + short(focus) + " %{y:.2%}<extra></extra>",
))
fig.add_trace(go.Scatter(
    x=line_x, y=line_x, mode="lines", name="β=1 기준선",
    line=dict(color=MUTED, width=1, dash="dot"), hoverinfo="skip",
))
fig.add_trace(go.Scatter(
    x=line_x, y=line_y, mode="lines", name=f"회귀선  β={r['beta']:.2f}",
    line=dict(color=CAT[1], width=2), hoverinfo="skip",
))
fig.update_xaxes(title_text=f"{label} 수익률", tickformat=".0%")
fig.update_yaxes(title_text=f"{short(focus)} 수익률", tickformat=".0%")
st.plotly_chart(style_fig(fig, height=420), use_container_width=True, key="scatter")
st.caption(
    f"점 하나 = {ctx.timeframe} 봉 하나. 점들이 회귀선에 바짝 붙을수록 R²가 큽니다 "
    f"(= {label}만 알면 이 코인이 예측된다)."
)


# ======================================================================
# 자동 해석
# ======================================================================

r2, beta = r["r2"], r["beta"]

if r2 >= 0.70:
    grade = f"{label} 추종형"
    grade_tone = f"{label}를 따로 들고 있다면 중복 노출이 큰 편입니다."
elif r2 >= 0.45:
    grade = f"{label} 영향 큼"
    grade_tone = f"방향성은 {label}가 많이 결정하고, 분산 효과는 제한적입니다."
elif r2 >= 0.25:
    grade = "혼합형"
    grade_tone = f"{label}도 중요하지만 코인 자체 움직임도 꽤 남아 있습니다."
else:
    grade = "독립성 높음"
    grade_tone = f"이 구간에서는 {label}만 보고 움직임을 설명하기 어렵습니다."

if beta >= 1.20:
    beta_plain = f"{label}보다 크게 출렁입니다"
elif beta <= 0.80:
    beta_plain = f"{label}보다 덜 출렁입니다"
else:
    beta_plain = f"{label}와 비슷한 폭으로 움직였습니다"

alpha_direction = ("좋아 보입니다" if r["alpha_ann"] > 0
                   else "나빠 보입니다" if r["alpha_ann"] < 0 else "중립입니다")
if r["p_alpha"] < 0.05:
    alpha_plain = f"{label} 영향을 빼고도 평균 성과가 {alpha_direction}. 비교적 믿을 만합니다"
else:
    alpha_plain = f"{label} 영향을 빼면 평균 성과가 {alpha_direction}. 다만 확실한 차이라고 보긴 어렵습니다"

st.markdown(f"#### {short(focus)} 한눈에 해석")

rows_html = [
    ("결론", grade, grade_tone),
    (f"{label} 의존도", f"{r2:.0%}", f"{r2:.0%}는 {label}, {1 - r2:.0%}는 자체 요인"),
    ("움직임 크기", f"beta {beta:.2f}", beta_plain),
    ("총 변동성", f"{r['total_vol']:.0%}", "1년 기준으로 환산한 가격 변동성 크기"),
    ("고유 리스크", f"{r['resid_vol']:.0%}", f"{label} 영향 제외 후 남는 코인 자체 변동성"),
    ("알파", f"{r['alpha_ann']:+.1%}", alpha_plain),
]

st.markdown(
    "<table class='coin-summary-table'>"
    "<thead><tr><th>항목</th><th>값</th><th>해석</th></tr></thead><tbody>"
    + "".join(f"<tr><td>{a}</td><td>{b}</td><td>{c}</td></tr>" for a, b, c in rows_html)
    + "</tbody></table>",
    unsafe_allow_html=True,
)
st.caption(f"표본 {r['n']:,}봉 · β 표준오차 {r['se_beta']:.3f} · t={r['t_zero']:.1f}")
