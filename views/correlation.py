"""상관관계 — 기준 코인 말고 자기들끼리는 어떤가.

기준 코인과의 관계(β·R²)만 보면 "각각이 얼마나 끌려다니는가"까지만 안다.
포트폴리오를 짜려면 코인들 사이 관계도 봐야 해서 행렬로 한 번에 놓는다.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from core.theme import DIVERGING, FONT, MUTED, short, style_fig
from core.ui import get_context, page_header

ctx = get_context()
page_header("🔗 상관관계", "기준 코인 말고 자기들끼리는", ctx)

label = ctx.market_label

cols = [ctx.market] + [m for m in ctx.alts if m in ctx.rets.columns]
corr = ctx.rets[cols].corr()
labels = [short(m) for m in corr.columns]

fig = go.Figure(go.Heatmap(
    z=corr.values, x=labels, y=labels,
    zmin=-1, zmax=1, zmid=0, colorscale=DIVERGING,
    xgap=2, ygap=2,
    text=np.round(corr.values, 2), texttemplate="%{text:.2f}",
    textfont=dict(size=11, family=FONT),
    colorbar=dict(title=dict(text="ρ", font=dict(color=MUTED, size=11)),
                  tickfont=dict(color=MUTED, size=10), thickness=12, outlinewidth=0),
    hovertemplate="%{y} ↔ %{x}<br>ρ = %{z:.3f}<extra></extra>",
))
fig.update_yaxes(autorange="reversed")
st.plotly_chart(style_fig(fig, height=460, legend=False), use_container_width=True, key="corr")

st.markdown(
    """
    | 색 | ρ 범위 | 의미 |
    |:---:|:---|:---|
    | 🔴 빨강 | 0.7 이상 | 거의 같이 움직임 |
    | ⬜ 회색 | 0 근처 | 관계 약함 |
    | 🔵 파랑 | 음수 | 반대로 움직임 |
    """
)


# ======================================================================
# 분산 효과 진단 — 행렬만 보면 "그래서 어쩌라고"가 남는다
# ======================================================================

st.markdown("#### 분산 효과가 있나")

alt_only = corr.drop(index=ctx.market, columns=ctx.market)
iu = np.triu_indices_from(alt_only.values, k=1)
pairs = alt_only.values[iu]

mean_rho = float(np.mean(pairs))
max_rho = float(np.max(pairs))
min_rho = float(np.min(pairs))

i_max, j_max = iu[0][int(np.argmax(pairs))], iu[1][int(np.argmax(pairs))]
i_min, j_min = iu[0][int(np.argmin(pairs))], iu[1][int(np.argmin(pairs))]
names = list(alt_only.columns)

c1, c2, c3 = st.columns(3)
c1.metric("알트끼리 평균 상관", f"{mean_rho:.2f}",
          help="기준 코인을 뺀 코인들 사이의 평균 상관계수")
c2.metric("가장 비슷한 쌍", f"{max_rho:.2f}",
          f"{short(names[i_max])} ↔ {short(names[j_max])}", delta_color="off")
c3.metric("가장 다른 쌍", f"{min_rho:.2f}",
          f"{short(names[i_min])} ↔ {short(names[j_min])}", delta_color="off")

# 여러 개를 담았을 때 실제로 몇 개를 담은 셈인지
n = len(names)
eff_n = n / (1 + (n - 1) * mean_rho) if (1 + (n - 1) * mean_rho) > 0 else float("nan")

st.markdown(
    f"""
    코인 **{n}개**를 같은 비중으로 담아도, 평균 상관이 **{mean_rho:.2f}** 라면
    분산 관점에서는 사실상 **{eff_n:.1f}개**를 담은 것과 같습니다.

    <sub>등비중 포트폴리오의 유효 종목 수 = n / (1 + (n−1)·ρ̄).
    상관이 0이면 n개 그대로, 상관이 1이면 1개로 수렴합니다.</sub>
    """,
    unsafe_allow_html=True,
)

if mean_rho >= 0.6:
    st.error(
        f"평균 상관이 {mean_rho:.2f}로 높습니다. **여러 개를 담아도 사실상 한 방향 베팅**에 가깝습니다. "
        f"{label} 하나만 들고 있는 것과 리스크 성격이 크게 다르지 않을 수 있습니다."
    )
elif mean_rho >= 0.4:
    st.warning(
        f"평균 상관 {mean_rho:.2f}. 분산 효과가 있긴 하지만 생각보다 작습니다. "
        f"특히 급락장에서는 이 값이 더 올라갑니다."
    )
else:
    st.info(f"평균 상관 {mean_rho:.2f}. 이 구간에서는 분산 효과가 어느 정도 남아 있습니다.")

st.caption(
    "알트끼리도 서로 빨갛다면 그건 대부분 **기준 코인이라는 공통 원인** 때문입니다. "
    "상관은 평시에 낮아도 급락장에서 1로 수렴하는 성질이 있어, 정작 분산이 필요한 순간에 "
    "효과가 사라집니다."
)
