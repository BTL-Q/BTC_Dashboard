"""전체 순위 — 누가 얼마나 기준 코인을 따라가나.

순위를 보고 곧바로 수치를 확인하는 흐름이라 막대와 전체 표를 한 페이지에 둔다.
"""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from core.theme import COIN_COLOR, INK_2, MUTED, short, style_fig
from core.ui import GLOSSARY, get_context, page_header

ctx = get_context()
page_header("📊 전체 순위", "누가 가장 기준 코인을 따라가나", ctx)

table, label = ctx.table, ctx.market_label


# ======================================================================
# 리더보드
# ======================================================================

follow_rank = table.sort_values(["r2", "beta"], ascending=False)
beta_rank = table.sort_values(["beta", "r2"], ascending=False)
top_follow, top_follow_m = follow_rank.iloc[0], str(follow_rank.index[0])
top_beta, top_beta_m = beta_rank.iloc[0], str(beta_rank.index[0])

st.markdown(
    f"""
    <div class="coin-card-grid">
      <div class="coin-card">
        <div class="coin-card-label">{label} 추종 1위 (R² 기준)</div>
        <div class="coin-card-name">{short(top_follow_m)}</div>
        <div class="coin-card-value">{top_follow["r2"]:.0%}</div>
        <div class="coin-card-sub">움직임의 {top_follow["r2"]:.0%}가 {label} 수익률로 설명됨</div>
      </div>
      <div class="coin-card">
        <div class="coin-card-label">{label} 1% 변동 시 반응</div>
        <div class="coin-card-name">{short(top_follow_m)}</div>
        <div class="coin-card-value">beta {top_follow["beta"]:.2f}</div>
        <div class="coin-card-sub">{label}가 1% 움직이면 평균 {top_follow["beta"]:.2f}% 움직인다는 뜻</div>
      </div>
      <div class="coin-card">
        <div class="coin-card-label">가장 민감한 코인 (beta 기준)</div>
        <div class="coin-card-name">{short(top_beta_m)}</div>
        <div class="coin-card-value">beta {top_beta["beta"]:.2f}</div>
        <div class="coin-card-sub">R² {top_beta["r2"]:.0%} · 상관 {top_beta["rho"]:.2f}</div>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

top3 = follow_rank.head(3)
st.caption(f"{label} \ucd94\uc885 TOP 3")
top3_cols = st.columns(3)
for column, market in zip(top3_cols, top3.index):
    column.markdown(f"**{short(str(market))}**")


# ======================================================================
# 순위 막대 — 설명력과 민감도는 다른 질문이라 나란히 둔다
# ======================================================================

c1, c2 = st.columns(2)

with c1:
    d = table.sort_values("r2")
    fig = go.Figure(go.Bar(
        x=d["r2"], y=[short(m) for m in d.index], orientation="h",
        marker=dict(color=[COIN_COLOR[m] for m in d.index]),
        text=[f"{v:.0%}" for v in d["r2"]], textposition="inside",
        insidetextanchor="end", textangle=0, cliponaxis=False,
        textfont=dict(color="#ffffff", size=13),
        hovertemplate=f"<b>%{{y}}</b><br>R² = %{{x:.3f}}<br>움직임의 %{{x:.0%}}가 {label}로 설명<extra></extra>",
        name="R²",
    ))
    fig.update_xaxes(range=[0, min(1.0, float(d["r2"].max()) * 1.25)], tickformat=".0%",
                     title_text=f"R² — {label}로 설명되는 비율")
    st.plotly_chart(style_fig(fig, height=360, legend=False), use_container_width=True, key="bar_r2")
    st.caption(f"클수록 {label} 영향이 큽니다. 이 코인 수익률 변동 중 {label}로 설명되는 몫입니다.")

with c2:
    d = table.sort_values("beta")
    fig = go.Figure(go.Bar(
        x=d["beta"], y=[short(m) for m in d.index], orientation="h",
        marker=dict(color=[COIN_COLOR[m] for m in d.index]),
        error_x=dict(type="data", symmetric=False,
                     array=(d["beta_hi"] - d["beta"]).values,
                     arrayminus=(d["beta"] - d["beta_lo"]).values,
                     color=MUTED, thickness=1, width=4),
        text=[f"{v:.2f}" for v in d["beta"]], textposition="inside",
        insidetextanchor="end", textangle=0, cliponaxis=False,
        textfont=dict(color="#ffffff", size=13),
        hovertemplate=f"<b>%{{y}}</b><br>β = %{{x:.3f}}<br>{label} 1%당 %{{x:.2f}}% 움직임<extra></extra>",
        name="β",
    ))
    fig.add_vline(x=1.0, line=dict(color=MUTED, width=1))
    fig.add_annotation(x=1.0, y=1.03, yref="paper", text=f"β=1 ({label}와 동일)",
                       showarrow=False, font=dict(color=MUTED, size=10), xanchor="left")
    fig.update_xaxes(title_text=f"β — {label} 1% 움직일 때의 반응 (막대 끝 = 95% 신뢰구간)")
    st.plotly_chart(style_fig(fig, height=360, legend=False), use_container_width=True, key="bar_beta")
    st.caption(f"1보다 크면 증폭입니다. {label}가 10% 빠질 때 beta=1.5면 평균 15% 빠진다는 뜻입니다.")


# ======================================================================
# 전체 수치 — 막대에서 본 순위를 숫자로 확인
# ======================================================================

st.markdown("### 전체 수치")

view = pd.DataFrame({
    "코인": [short(m) for m in table.index],
    "β": table["beta"],
    "β 95% CI": [f"[{lo:.2f}, {hi:.2f}]" for lo, hi in zip(table["beta_lo"], table["beta_hi"])],
    "SE(β)": table["se_beta"],
    "β=1 p값": table["p_one"],
    "R²": table["r2"] * 100,
    "ρ": table["rho"],
    "α (연율)": table["alpha_ann"] * 100,
    "α p값": table["p_alpha"],
    "고유변동성": table["resid_vol"] * 100,
    "총변동성": table["total_vol"] * 100,
    "표본": table["n"].astype(int),
}).sort_values("R²", ascending=False).reset_index(drop=True)

st.dataframe(
    view, use_container_width=True, hide_index=True,
    column_config={
        "β": st.column_config.NumberColumn(format="%.3f", help=GLOSSARY["beta"]),
        "SE(β)": st.column_config.NumberColumn(format="%.3f", help=GLOSSARY["se"]),
        "β=1 p값": st.column_config.NumberColumn(format="%.3f", help=GLOSSARY["p_one"]),
        "R²": st.column_config.NumberColumn(format="%.1f%%", help=GLOSSARY["r2"]),
        "ρ": st.column_config.NumberColumn(format="%.2f", help=GLOSSARY["rho"]),
        "α (연율)": st.column_config.NumberColumn(format="%+.1f%%", help=GLOSSARY["alpha"]),
        "α p값": st.column_config.NumberColumn(format="%.3f"),
        "고유변동성": st.column_config.NumberColumn(format="%.1f%%", help=GLOSSARY["resid_vol"]),
        "총변동성": st.column_config.NumberColumn(format="%.1f%%", help=GLOSSARY["total_vol"]),
        "표본": st.column_config.NumberColumn(format="%d"),
    },
)

st.download_button(
    "CSV 내려받기", view.to_csv(index=False).encode("utf-8-sig"),
    file_name=f"{label.lower()}_beta_{ctx.timeframe}_{ctx.period_label}.csv",
    mime="text/csv",
)
