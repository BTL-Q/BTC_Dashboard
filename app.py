"""코인 영향력 분석 (Market Model) — Streamlit 페이지.

"이 코인이 기준 코인에 얼마나 끌려다니는가"를 숫자로 표현한다.
시장(market) 자리에 기준 코인을 놓고 다른 코인의 수익률을 회귀시키는 시장모형:

    r_coin,t = α + β · r_base,t + ε_t

- β   : 기준 코인이 1% 움직일 때 이 코인이 몇 % 움직이나 (민감도 · 증폭률)
- R²  : 이 코인 움직임 중 기준 코인으로 설명되는 비율 → "영향력"의 직접적인 답
- α   : 기준 코인으로 설명되지 않는 고유 수익 (연율화)
- σ_ε : 기준 코인 영향을 제거하고 남은 이 코인만의 변동성 (연율화)

기준 코인은 페이지 상단에서 선택 (BTC / ETH / SOL / XRP).

실행: streamlit run app.py
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from scipy import stats

from core.config import (
    BARS_PER_YEAR,
    BASE_CANDIDATES,
    BUNDLED_TIMEFRAMES,
    DEFAULT_COINS,
    OHLCV_DIR,
)
from core.data import load_ohlcv


# ======================================================================
# 페이지 설정
# ======================================================================

st.set_page_config(
    page_title="코인 영향력",
    page_icon="🧲",
    layout="wide",
)

st.title("🧲 코인 영향력 분석")
st.caption("시장모형 회귀 `r_coin = α + β·r_base + ε` — 선택한 기준 코인에 다른 코인이 얼마나 끌려다니는지 숫자로")


# ======================================================================
# 색상 — dataviz 검증 팔레트 (validate_palette.js 통과)
#   light: 8슬롯 전부 PASS (contrast WARN → 범례 + 표 뷰로 보완)
#   dark : 8슬롯 전부 PASS
# 색은 "코인(엔티티)"에 고정한다. 정렬 순서나 필터로 절대 바뀌지 않음.
# ======================================================================

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

BASE_MARKETS = [m for m in BASE_CANDIDATES if m in DEFAULT_COINS]
if not BASE_MARKETS:
    BASE_MARKETS = [DEFAULT_COINS[0]]

# 코인 → 고정 색 (엔티티 색, 순위 무관)
COIN_COLOR = {m: CAT[i % len(CAT)] for i, m in enumerate(DEFAULT_COINS)}

# 상관행렬용 발산형 스케일: 파랑 ← 중립 회색 → 빨강 (중앙 0 = "관계 없음")
DIVERGING = [
    [0.00, "#0d366b"],
    [0.25, "#2a78d6" if not DARK else "#3987e5"],
    [0.50, NEUTRAL_MID],
    [0.75, "#e34948" if not DARK else "#e66767"],
    [1.00, "#7f1d1d"],
]


def short(market: str) -> str:
    """KRW-BTC → BTC"""
    return market.split("-")[-1]


# ======================================================================
# 용어 사전 — 기존 '전략 실험실' 페이지와 같은 컨벤션
# ======================================================================

GLOSSARY = {
    "β (베타)": "회귀직선의 기울기. 기준 코인이 1% 오를 때 이 코인이 평균 몇 % 오르는가. "
                "β=1이면 기준 코인과 같은 폭, β=1.5면 1.5배 증폭, β=0.5면 절반만 반응.",
    "R² (결정계수)": "이 코인 수익률 변동 중 기준 코인으로 설명되는 비율(0~1). "
                     "0.8이면 움직임의 80%가 기준 코인 영향. '영향력'을 재는 핵심 숫자.",
    "α (알파)": "기준 코인으로 설명되지 않는 고유 수익. 연율화해서 표시. "
                "양수면 기준 코인 대비 초과수익이지만, 통계적 유의성(p)을 같이 봐야 함.",
    "ρ (상관계수)": "두 수익률이 같이 움직인 정도(-1~1). 단순회귀에서는 ρ² = R².",
    "표준오차 (SE)": "β 추정치의 불확실성. 작을수록 β를 믿을 만함.",
    "t값 / p값": "추정된 β가 우연일 확률 검정. p < 0.05면 '우연이 아니다'라고 본다.",
    "β=1 검정": "β가 1과 통계적으로 다른가. p < 0.05여야 '진짜 증폭/둔감'이라 말할 수 있음.",
    "잔차 변동성 (σ_ε)": "기준 코인 영향을 빼고 남은 이 코인만의 변동성(연율화). "
                         "= 분산투자로 줄일 수 있는 고유 리스크.",
    "총 변동성": "이 코인 수익률의 전체 변동성(연율화). 기준 코인 부분 + 고유 부분.",
    "롤링 β": "고정 구간이 아니라 최근 N봉만 써서 계속 다시 계산한 β. "
              "시간이 갈수록 기준 코인의 지배력이 세지는지 약해지는지 보여줌.",
    "로그수익률": "log(P_t / P_t-1). 덧셈으로 누적되어 회귀·연율화에 적합. 회귀 기본값.",
    "단순수익률": "P_t / P_t-1 - 1. 직관적이지만 큰 변동에서 비대칭.",
    "공통 구간": "선택한 코인 전부가 데이터를 가진 기간만 사용. "
                 "상장일이 다른 코인끼리 공정하게 비교하려면 필요.",
}


def help_text(term: str) -> str:
    return GLOSSARY.get(term, "")


with st.sidebar:
    with st.expander("📖 용어 사전", expanded=False):
        for term, desc in GLOSSARY.items():
            st.markdown(f"**{term}** — {desc}")


# ======================================================================
# 데이터
# ======================================================================


@st.cache_data(show_spinner=False)
def load_close(market: str, timeframe: str) -> pd.Series:
    """종가만 읽기. 동봉 parquet 우선, 없으면 Upbit API."""
    df = load_ohlcv(market, timeframe)
    if df.empty:
        return pd.Series(dtype=float)
    return df["close"].astype(float)


@st.cache_data(show_spinner=False)
def build_returns(markets: tuple[str, ...], timeframe: str, use_log: bool) -> pd.DataFrame:
    """코인별 종가 → 수익률 DataFrame (열=코인)."""
    closes = {}
    for m in markets:
        s = load_close(m, timeframe)
        if not s.empty:
            closes[m] = s
    if not closes:
        return pd.DataFrame()
    px = pd.DataFrame(closes).sort_index()
    px = px[~px.index.duplicated(keep="last")]
    px = px.where(px > 0)
    rets = np.log(px).diff() if use_log else px.pct_change()
    return rets


# ======================================================================
# 회귀 — 단순 OLS를 직접 계산 (statsmodels 미설치 환경)
# ======================================================================


def market_model(y: np.ndarray, x: np.ndarray, bars_per_year: float) -> dict | None:
    """y = α + β·x + ε 단순회귀. y=비교 코인 수익률, x=기준 코인 수익률."""
    n = len(x)
    if n < 30:
        return None

    xbar, ybar = float(x.mean()), float(y.mean())
    dx, dy = x - xbar, y - ybar
    sxx = float((dx * dx).sum())
    sst = float((dy * dy).sum())
    if sxx <= 0 or sst <= 0:
        return None

    beta = float((dx * dy).sum() / sxx)
    alpha = ybar - beta * xbar
    resid = y - (alpha + beta * x)
    sse = float((resid * resid).sum())

    dof = n - 2
    s2 = sse / dof
    se_beta = math.sqrt(s2 / sxx)
    se_alpha = math.sqrt(s2 * (1.0 / n + xbar * xbar / sxx))

    t_zero = beta / se_beta if se_beta > 0 else np.nan          # H0: β = 0 (관계 없음)
    t_one = (beta - 1.0) / se_beta if se_beta > 0 else np.nan   # H0: β = 1 (기준 코인과 동일)
    crit = float(stats.t.ppf(0.975, dof))

    return {
        "beta": beta,
        "se_beta": se_beta,
        "beta_lo": beta - crit * se_beta,
        "beta_hi": beta + crit * se_beta,
        "t_zero": float(t_zero),
        "p_zero": float(2 * stats.t.sf(abs(t_zero), dof)),
        "t_one": float(t_one),
        "p_one": float(2 * stats.t.sf(abs(t_one), dof)),
        "alpha_bar": alpha,
        "alpha_ann": alpha * bars_per_year,
        "se_alpha": se_alpha,
        "p_alpha": float(2 * stats.t.sf(abs(alpha / se_alpha), dof)) if se_alpha > 0 else np.nan,
        "r2": 1.0 - sse / sst,
        "rho": float(np.corrcoef(x, y)[0, 1]),
        "resid_vol": float(resid.std(ddof=1)) * math.sqrt(bars_per_year),
        "total_vol": float(y.std(ddof=1)) * math.sqrt(bars_per_year),
        "n": n,
    }


def rolling_beta_r2(y: pd.Series, x: pd.Series, window: int):
    """단순회귀 β = cov(x,y)/var(x), R² = corr². 롤링으로 한 번에."""
    beta = y.rolling(window).cov(x) / x.rolling(window).var()
    r2 = y.rolling(window).corr(x) ** 2
    return beta, r2


# ======================================================================
# 차트 공통 스타일
# ======================================================================


def style_fig(fig: go.Figure, height: int = 380, legend: bool = True,
              hovermode: str = "closest") -> go.Figure:
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


# ======================================================================
# 필터 — 아래 모든 차트를 같은 슬라이스로 스코프
# ======================================================================

f0, f1, f2, f3, f4 = st.columns([1, 1, 1.4, 1, 1])

with f0:
    MARKET = st.selectbox("기준 코인", BASE_MARKETS, index=0, format_func=short)
    market_label = short(MARKET)
    ALTS = [c for c in DEFAULT_COINS if c != MARKET]
with f1:
    timeframe = st.selectbox("봉", ["1d", "1h"], index=0,
                             help="1d = 일봉(구조적 관계), 1h = 시간봉(표본 많음·노이즈 큼)")
with f3:
    ret_kind = st.radio("수익률", ["로그", "단순"], index=0, horizontal=True,
                        help=help_text("로그수익률"))
with f4:
    common_span = st.toggle("공통 구간", value=True, help=help_text("공통 구간"))

use_log = ret_kind == "로그"
bars_per_year = BARS_PER_YEAR[timeframe]

rets_all = build_returns(tuple(DEFAULT_COINS), timeframe, use_log)

if rets_all.empty or MARKET not in rets_all.columns:
    if timeframe in BUNDLED_TIMEFRAMES:
        st.error(f"데이터를 찾을 수 없습니다. `data/ohlcv/` 를 확인하세요. ({OHLCV_DIR})")
    else:
        st.warning(
            f"**{timeframe} 데이터를 불러오지 못했습니다.**\n\n"
            "이 저장소에는 일봉(1d)만 동봉되어 있고, 나머지 타임프레임은 실행 시 "
            "Upbit API로 받아옵니다. Upbit이 해외 IP를 차단하는 경우가 있어 "
            "클라우드 배포 환경에서는 실패할 수 있습니다."
        )
        st.info("위 **봉** 선택에서 **1d**를 고르시면 동봉된 데이터로 바로 보실 수 있습니다.")
    st.stop()

data_start = rets_all.index.min()
data_end = rets_all.index.max()

PERIODS = {
    "전체": None,
    "최근 3년": 365 * 3,
    "최근 2년": 365 * 2,
    "최근 1년": 365,
    "최근 6개월": 182,
    "최근 3개월": 91,
}
with f2:
    period_label = st.selectbox("기간", list(PERIODS.keys()), index=3)

days = PERIODS[period_label]
if days is None:
    window_start = data_start
else:
    window_start = data_end - pd.Timedelta(days=days)

rets = rets_all[rets_all.index >= window_start]

# 상장일이 다른 코인 정렬
if common_span:
    rets = rets.dropna(how="any")
else:
    rets = rets.dropna(subset=[MARKET])

stale_days = (pd.Timestamp.now(tz="UTC") - data_end).days
freshness = f" · 데이터 스냅샷 기준 (마지막 봉 {stale_days}일 전)" if stale_days > 3 else ""
st.caption(
    f"보유 데이터 {data_start.date()} ~ {data_end.date()} · "
    f"분석 구간 **{rets.index.min().date()} ~ {rets.index.max().date()}** · "
    f"{len(rets):,}봉 · 기준 = {market_label}{freshness}"
)

if len(rets) < 30:
    st.warning("표본이 30봉 미만입니다. 기간을 늘리거나 '공통 구간'을 끄세요.")
    st.stop()


# ======================================================================
# 전체 코인 회귀
# ======================================================================

results: dict[str, dict] = {}
for m in ALTS:
    if m not in rets.columns:
        continue
    pair = rets[[MARKET, m]].dropna()
    if len(pair) < 30:
        continue
    res = market_model(pair[m].values, pair[MARKET].values, bars_per_year)
    if res:
        results[m] = res

if not results:
    st.error("회귀 가능한 코인이 없습니다.")
    st.stop()

table = pd.DataFrame(results).T
table.index.name = "market"

st.markdown(
    """
    <style>
    .coin-leader-grid {
        display: grid;
        grid-template-columns: repeat(3, minmax(0, 1fr));
        gap: 12px;
        margin: 10px 0 18px;
    }
    .coin-leader-card {
        border: 1px solid rgba(128, 128, 128, 0.22);
        border-radius: 8px;
        padding: 14px 16px;
        background: rgba(128, 128, 128, 0.06);
    }
    .coin-leader-label {
        color: rgba(128, 128, 128, 0.95);
        font-size: 13px;
        line-height: 1.2;
        margin-bottom: 8px;
    }
    .coin-leader-name {
        font-size: 28px;
        font-weight: 750;
        line-height: 1.05;
        margin-bottom: 10px;
    }
    .coin-leader-value {
        font-size: 34px;
        font-weight: 800;
        line-height: 1;
        letter-spacing: 0;
    }
    .coin-leader-sub {
        color: rgba(128, 128, 128, 0.95);
        font-size: 13px;
        margin-top: 8px;
        line-height: 1.35;
    }
    @media (max-width: 900px) {
        .coin-leader-grid {
            grid-template-columns: 1fr;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

follow_rank = table.sort_values(["r2", "beta"], ascending=False)
beta_rank = table.sort_values(["beta", "r2"], ascending=False)
top_follow = follow_rank.iloc[0]
top_follow_market = str(follow_rank.index[0])
top_beta = beta_rank.iloc[0]
top_beta_market = str(beta_rank.index[0])

top3 = follow_rank.head(3)
top3_line = " · ".join(
    f"{short(str(m))} {row['r2']:.0%} / beta {row['beta']:.2f}"
    for m, row in top3.iterrows()
)

st.markdown(f"### 지금 {market_label}를 가장 따라가는 코인")
st.markdown(
    f"""
    <div class="coin-leader-grid">
      <div class="coin-leader-card">
        <div class="coin-leader-label">{market_label} 추종 1위 (R² 기준)</div>
        <div class="coin-leader-name">{short(top_follow_market)}</div>
        <div class="coin-leader-value">{top_follow["r2"]:.0%}</div>
        <div class="coin-leader-sub">움직임의 {top_follow["r2"]:.0%}가 {market_label} 수익률로 설명됨</div>
      </div>
      <div class="coin-leader-card">
        <div class="coin-leader-label">{market_label} 1% 변동 시 반응</div>
        <div class="coin-leader-name">{short(top_follow_market)}</div>
        <div class="coin-leader-value">beta {top_follow["beta"]:.2f}</div>
        <div class="coin-leader-sub">{market_label}가 1% 움직이면 평균 {top_follow["beta"]:.2f}% 움직인다는 뜻</div>
      </div>
      <div class="coin-leader-card">
        <div class="coin-leader-label">가장 민감한 코인 (beta 기준)</div>
        <div class="coin-leader-name">{short(top_beta_market)}</div>
        <div class="coin-leader-value">beta {top_beta["beta"]:.2f}</div>
        <div class="coin-leader-sub">R² {top_beta["r2"]:.0%} · 상관 {top_beta["rho"]:.2f}</div>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)
st.caption(f"{market_label} 추종 TOP 3: {top3_line}")


# ======================================================================
# 1. 한눈에 — R² 순위 / β 순위
# ======================================================================

st.markdown(f"### 1. 한눈에 — 누가 가장 {market_label}를 따라가나")

c1, c2 = st.columns(2)

with c1:
    d = table.sort_values("r2")
    fig = go.Figure(go.Bar(
        x=d["r2"], y=[short(m) for m in d.index], orientation="h",
        marker=dict(color=[COIN_COLOR[m] for m in d.index]),
        text=[f"{v:.0%}" for v in d["r2"]], textposition="inside",
        textfont=dict(color="#ffffff", size=13),
        insidetextanchor="end",
        textangle=0,
        cliponaxis=False,
        hovertemplate=f"<b>%{{y}}</b><br>R² = %{{x:.3f}}<br>움직임의 %{{x:.0%}}가 {market_label}로 설명<extra></extra>",
        name="R²",
    ))
    fig.update_xaxes(range=[0, min(1.0, float(d["r2"].max()) * 1.25)], tickformat=".0%",
                     title_text=f"R² — {market_label}로 설명되는 비율")
    st.plotly_chart(style_fig(fig, height=360, legend=False), use_container_width=True,
                    key="bar_r2")
    st.caption(f"클수록 {market_label} 영향이 큽니다. 이 코인 수익률 변동 중 {market_label}로 설명되는 몫입니다.")

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
        textfont=dict(color="#ffffff", size=13),
        insidetextanchor="end",
        textangle=0,
        cliponaxis=False,
        hovertemplate=f"<b>%{{y}}</b><br>β = %{{x:.3f}}<br>{market_label} 1%당 %{{x:.2f}}% 움직임<extra></extra>",
        name="β",
    ))
    fig.add_vline(x=1.0, line=dict(color=MUTED, width=1))
    fig.add_annotation(x=1.0, y=1.03, yref="paper", text=f"β=1 ({market_label}와 동일)",
                       showarrow=False, font=dict(color=MUTED, size=10), xanchor="left")
    fig.update_xaxes(title_text=f"β — {market_label} 1% 움직일 때의 반응 (막대 끝 = 95% 신뢰구간)")
    st.plotly_chart(style_fig(fig, height=360, legend=False), use_container_width=True,
                    key="bar_beta")
    st.caption(f"1보다 크면 증폭입니다. {market_label}가 10% 빠질 때 beta=1.5면 평균 15% 빠진다는 뜻입니다.")


# ======================================================================
# 2. 코인 하나 파고들기
# ======================================================================

st.markdown("### 2. 코인 하나 파고들기")

focus = st.selectbox(
    "코인", list(results.keys()), index=0,
    format_func=short, key="focus_coin",
)
r = results[focus]

m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("β (민감도)", f"{r['beta']:.2f}",
          f"±{(r['beta_hi'] - r['beta']):.2f} (95% CI)", delta_color="off",
          help=help_text("β (베타)"))
m2.metric("R² (설명력)", f"{r['r2']:.1%}", help=help_text("R² (결정계수)"))
m3.metric("ρ (상관)", f"{r['rho']:.2f}", help=help_text("ρ (상관계수)"))
m4.metric("α (연율)", f"{r['alpha_ann']:+.1%}",
          "유의" if r["p_alpha"] < 0.05 else "무의미", delta_color="off",
          help=help_text("α (알파)"))
m5.metric("고유 변동성", f"{r['resid_vol']:.1%}",
          f"총 {r['total_vol']:.0%} 중", delta_color="off",
          help=help_text("잔차 변동성 (σ_ε)"))

with st.container():
    pair = rets[[MARKET, focus]].dropna()
    xs, ys = pair[MARKET].values, pair[focus].values
    line_x = np.array([xs.min(), xs.max()])
    line_y = r["alpha_bar"] + r["beta"] * line_x

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=xs, y=ys, mode="markers", name=f"{short(focus)} 봉별 수익률",
        marker=dict(color=COIN_COLOR[focus], size=7, opacity=0.40,
                    line=dict(width=0)),
        hovertemplate=f"{market_label} %{{x:.2%}}<br>" + short(focus) + " %{y:.2%}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=line_x, y=line_x, mode="lines", name="β=1 기준선",
        line=dict(color=MUTED, width=1, dash="dot"), hoverinfo="skip",
    ))
    fig.add_trace(go.Scatter(
        x=line_x, y=line_y, mode="lines",
        name=f"회귀선  β={r['beta']:.2f}",
        line=dict(color=CAT[1], width=2), hoverinfo="skip",
    ))
    fig.update_xaxes(title_text=f"{market_label} 수익률", tickformat=".0%")
    fig.update_yaxes(title_text=f"{short(focus)} 수익률", tickformat=".0%")
    st.plotly_chart(style_fig(fig, height=380), use_container_width=True, key="scatter")
    st.caption(
        f"점 하나 = {timeframe} 봉 하나. 점들이 회귀선에 바짝 붙을수록 R²가 크다 "
        f"(= {market_label}만 알면 이 코인이 예측된다)."
    )

with st.container():
    # 자동 해석
    r2, beta = r["r2"], r["beta"]
    if r2 >= 0.70:
        grade = f"{market_label} 추종형"
        grade_tone = f"{market_label}를 따로 들고 있다면 중복 노출이 큰 편입니다."
    elif r2 >= 0.45:
        grade = f"{market_label} 영향 큼"
        grade_tone = f"방향성은 {market_label}가 많이 결정하고, 분산 효과는 제한적입니다."
    elif r2 >= 0.25:
        grade = "혼합형"
        grade_tone = f"{market_label}도 중요하지만 코인 자체 움직임도 꽤 남아 있습니다."
    else:
        grade = "독립성 높음"
        grade_tone = f"이 구간에서는 {market_label}만 보고 움직임을 설명하기 어렵습니다."

    if beta >= 1.20:
        beta_plain = f"{market_label}보다 크게 출렁입니다. {market_label}가 1% 움직일 때 평균 {beta:.2f}% 움직였습니다."
    elif beta <= 0.80:
        beta_plain = f"{market_label}보다 덜 출렁입니다. {market_label}가 1% 움직일 때 평균 {beta:.2f}% 움직였습니다."
    else:
        beta_plain = f"{market_label}와 비슷한 폭으로 움직였습니다. {market_label}가 1% 움직일 때 평균 {beta:.2f}% 움직였습니다."

    if r["alpha_ann"] > 0:
        alpha_direction = "좋아 보입니다"
    elif r["alpha_ann"] < 0:
        alpha_direction = "나빠 보입니다"
    else:
        alpha_direction = "중립입니다"

    if r["p_alpha"] < 0.05:
        alpha_plain = f"{market_label} 영향을 빼고도 평균 성과가 {alpha_direction}. 이 차이는 비교적 믿을 만합니다."
    else:
        alpha_plain = f"{market_label} 영향을 빼면 평균 성과가 {alpha_direction}. 다만 확실한 차이라고 보긴 어렵습니다."

    st.markdown(f"#### {short(focus)} 한눈에 해석")
    rows_html = [
        ("결론", grade, grade_tone),
        (f"{market_label} 의존도", f"{r2:.0%}", f"{r2:.0%}는 {market_label}, {1 - r2:.0%}는 자체 요인"),
        ("움직임 크기", f"beta {beta:.2f}", beta_plain.replace(" 움직였습니다.", "")),
        ("총 변동성", f"{r['total_vol']:.0%}", "1년 기준으로 환산한 가격 변동성 크기"),
        ("고유 리스크", f"{r['resid_vol']:.0%}", f"{market_label} 영향 제외 후 남는 코인 자체 변동성"),
        ("알파", f"{r['alpha_ann']:+.1%}", alpha_plain.split(". ")[0]),
    ]
    st.markdown(
        """
        <style>
        .coin-summary-table {
            width: 100%;
            border-collapse: collapse;
            table-layout: fixed;
            font-size: 14px;
            line-height: 1.3;
        }
        .coin-summary-table th,
        .coin-summary-table td {
            border-bottom: 1px solid rgba(128, 128, 128, 0.22);
            padding: 9px 8px;
            vertical-align: top;
            word-break: keep-all;
            overflow-wrap: anywhere;
        }
        .coin-summary-table th {
            color: rgba(128, 128, 128, 0.95);
            font-weight: 650;
            text-align: left;
        }
        .coin-summary-table td:nth-child(1) {
            width: 16%;
            color: rgba(128, 128, 128, 0.95);
        }
        .coin-summary-table td:nth-child(2) {
            width: 18%;
            font-weight: 750;
            white-space: nowrap;
        }
        .coin-summary-table td:nth-child(3) {
            width: 66%;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        "<table class='coin-summary-table'>"
        "<thead><tr><th>항목</th><th>값</th><th>해석</th></tr></thead>"
        "<tbody>"
        + "".join(
            f"<tr><td>{label}</td><td>{value}</td><td>{desc}</td></tr>"
            for label, value, desc in rows_html
        )
        + "</tbody></table>",
        unsafe_allow_html=True,
    )
    st.caption(f"표본 {r['n']:,}봉 · β 표준오차 {r['se_beta']:.3f} · t={r['t_zero']:.1f}")


# ======================================================================
# 3. 시간에 따른 변화 — 롤링 β / 롤링 R²
# ======================================================================

st.markdown("### 3. 시간에 따라 변하나 — 롤링 회귀")

rc1, rc2 = st.columns([2.4, 1])
with rc1:
    picked = st.multiselect(
        "코인 (최대 8개)", list(results.keys()),
        default=list(results.keys())[:4], format_func=short, key="roll_coins",
    )
with rc2:
    default_win = 90 if timeframe == "1d" else 720
    max_win = max(60, min(len(rets) // 2, 1500))
    window = st.slider("롤링 윈도우 (봉)", 30, max_win,
                       min(default_win, max_win), step=10, help=help_text("롤링 β"))

if picked:
    beta_lines, r2_lines = {}, {}
    for m in picked:
        pair = rets[[MARKET, m]].dropna()
        b, rr = rolling_beta_r2(pair[m], pair[MARKET], window)
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
        # 계열 4개 이하면 끝점 직접 라벨 (색만으로 식별하지 않게)
        if len(series_map) <= 4:
            for m, s in series_map.items():
                if len(s):
                    fig.add_annotation(
                        x=s.index[-1], y=float(s.iloc[-1]), text=f" {short(m)}",
                        showarrow=False, xanchor="left",
                        font=dict(color=COIN_COLOR[m], size=11),
                    )
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
        | 1보다 위 | {market_label}보다 더 크게 움직이는 구간 |
        | 1 근처 | {market_label}와 비슷한 폭으로 움직이는 구간 |
        | 1보다 아래 | {market_label}보다 덜 움직이는 구간 |
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
        | 위로 올라감 | {market_label} 영향력이 커지는 구간 |
        | 아래로 내려감 | 코인 자체 움직임이 커지는 구간 |
        | 여러 코인이 동시에 높아짐 | 시장이 한 방향으로 묶이는 구간. 분산 효과가 줄어듦 |
        | beta는 높은데 R²는 낮음 | 크게 반응하긴 하지만, {market_label}를 꾸준히 따라간다고 보긴 어려움 |
        """
    )
    st.caption("롤링 회귀는 과거 전체 평균이 아니라, 지금 관계가 예전과 같은지 보는 용도입니다.")
else:
    st.info("코인을 하나 이상 선택하세요.")


# ======================================================================
# 4. 상관행렬
# ======================================================================

st.markdown(f"### 4. 상관행렬 — {market_label} 말고 자기들끼리는?")

corr_cols = [MARKET] + [m for m in ALTS if m in rets.columns]
corr = rets[corr_cols].corr()
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
    f"""
    상관행렬 해석  
    &nbsp;&nbsp;빨강 / 0.7 이상: 같이 움직임  
    &nbsp;&nbsp;회색 / 0 근처: 관계 약함  
    &nbsp;&nbsp;파랑 / 음수: 반대로 움직임  
    &nbsp;&nbsp;빨간 코인을 여러 개 담으면 분산 효과가 작을 수 있음
    """
)


# ======================================================================
# 5. 전체 수치 (표 뷰)
# ======================================================================

st.markdown("### 5. 전체 수치")

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
        "β": st.column_config.NumberColumn(format="%.3f", help=help_text("β (베타)")),
        "SE(β)": st.column_config.NumberColumn(format="%.3f", help=help_text("표준오차 (SE)")),
        "β=1 p값": st.column_config.NumberColumn(format="%.3f", help=help_text("β=1 검정")),
        "R²": st.column_config.NumberColumn(format="%.1f%%", help=help_text("R² (결정계수)")),
        "ρ": st.column_config.NumberColumn(format="%.2f", help=help_text("ρ (상관계수)")),
        "α (연율)": st.column_config.NumberColumn(format="%+.1f%%", help=help_text("α (알파)")),
        "α p값": st.column_config.NumberColumn(format="%.3f"),
        "고유변동성": st.column_config.NumberColumn(format="%.1f%%", help=help_text("잔차 변동성 (σ_ε)")),
        "총변동성": st.column_config.NumberColumn(format="%.1f%%", help=help_text("총 변동성")),
        "표본": st.column_config.NumberColumn(format="%d"),
    },
)

st.download_button(
    "CSV 내려받기", view.to_csv(index=False).encode("utf-8-sig"),
    file_name=f"{market_label.lower()}_beta_{timeframe}_{period_label}.csv", mime="text/csv",
)


# ======================================================================
# 함정 노트
# ======================================================================

with st.expander("⚠️ 투자에 쓸 때 체크할 것 5가지"):
    st.markdown(
        """
**1. 분산투자 확인.** 상관이나 R²가 높은 코인끼리는 여러 개 담아도 리스크가 많이 겹칩니다.
분산 목적이라면 서로 덜 빨갛고, 기준 코인 의존도가 낮은 코인을 같이 봐야 합니다.

**2. β가 큰 코인은 양날입니다.** 기준 코인이 오르면 더 크게 오를 수 있지만,
기준 코인이 빠지면 더 크게 빠질 수 있습니다. 상승 레버리지이자 하락 레버리지입니다.

**3. R²가 높으면 기준 코인 베팅에 가깝습니다.** 방향성을 기준 코인이 많이 결정합니다.
기준 코인을 이미 들고 있다면 중복 노출인지 확인해야 합니다.

**4. β와 R²는 고정값이 아닙니다.** 구간을 바꾸면 숫자가 달라집니다.
롤링 차트로 지금 관계가 강해지는지 약해지는지 같이 보세요.

**5. 같은 봉 회귀는 예측이 아니다.** 여기서는 *같은 시점*의 기준 코인과 비교 코인을 맞춥니다.
"기준 코인이 오르는 걸 보고 다른 코인을 산다"가 되려면 x를 한 봉 밀어야(shift) 하고,
그렇게 하면 R²는 대개 0에 가깝게 무너집니다.
        """
    )
