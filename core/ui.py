"""페이지 공통 — 사이드바 전역 필터와 데이터 준비.

모든 페이지가 첫 줄에서 `ctx = load_context()` 를 부른다.
필터는 사이드바에 한 번만 그려지고 위젯 key로 세션에 남으므로,
페이지를 옮겨다녀도 기준 코인·기간 설정이 그대로 유지된다.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import streamlit as st

from core.config import (
    BARS_PER_YEAR,
    BASE_CANDIDATES,
    BUNDLED_TIMEFRAMES,
    DEFAULT_COINS,
    OHLCV_DIR,
    UPBIT_MAX_FETCH_BARS,
)
from core.data import load_ohlcv
from core.stats import market_model
from core.theme import CARD_CSS, short

# 타임프레임별 하루치 봉 수 — 기간 선택에서 필요한 봉 수를 역산할 때 쓴다.
BARS_PER_DAY: dict[str, int] = {"1d": 1, "1h": 24}

PERIODS: dict[str, int | None] = {
    "전체": None,
    "최근 3년": 365 * 3,
    "최근 2년": 365 * 2,
    "최근 1년": 365,
    "최근 6개월": 182,
    "최근 3개월": 91,
}

# 호버 툴팁용 용어 설명. 사이드바 패널은 없애고 툴팁으로만 남긴다.
GLOSSARY = {
    "beta": "회귀직선의 기울기. 기준 코인이 1% 오를 때 이 코인이 평균 몇 % 오르는가. "
            "β=1이면 같은 폭, β=1.5면 1.5배 증폭, β=0.5면 절반만 반응.",
    "r2": "이 코인 수익률 변동 중 기준 코인으로 설명되는 비율(0~1). "
          "0.8이면 움직임의 80%가 기준 코인 영향.",
    "alpha": "기준 코인으로 설명되지 않는 고유 수익. 연율화해서 표시. "
             "통계적 유의성(p)을 같이 봐야 함.",
    "rho": "두 수익률이 같이 움직인 정도(-1~1). 단순회귀에서는 ρ² = R².",
    "se": "추정치의 불확실성. 작을수록 믿을 만함.",
    "p_one": "β가 1과 통계적으로 다른가. p < 0.05여야 '진짜 증폭/둔감'이라 말할 수 있음.",
    "resid_vol": "기준 코인 영향을 빼고 남은 이 코인만의 변동성(연율화). "
                 "분산투자로 줄일 수 있는 고유 리스크.",
    "total_vol": "이 코인 수익률의 전체 변동성(연율화). 기준 코인 부분 + 고유 부분.",
    "rolling": "고정 구간이 아니라 최근 N봉만 써서 계속 다시 계산한 값. "
               "관계가 시간에 따라 얼마나 흔들리는지 보여줌.",
    "log_return": "log(P_t / P_t-1). 덧셈으로 누적되어 회귀·연율화에 적합. 기본값.",
    "common_span": "선택한 코인 전부가 데이터를 가진 기간만 사용. "
                   "상장일이 다른 코인끼리 공정하게 비교하려면 필요.",
    "timeframe": "1d = 일봉(구조적 관계), 1h = 시간봉(표본 많음·노이즈 큼). "
                 "주기를 짧게 자를수록 측정되는 상관이 깎입니다(Epps effect).",
}


@dataclass
class Context:
    market: str
    market_label: str
    alts: list[str]
    timeframe: str
    period_label: str
    use_log: bool
    common_span: bool
    bars_per_year: float
    rets: pd.DataFrame
    results: dict[str, dict]
    table: pd.DataFrame
    data_start: pd.Timestamp
    data_end: pd.Timestamp
    coins: list[str] = field(default_factory=list)


@st.cache_data(show_spinner=False)
def _load_close(market: str, timeframe: str, bars: int) -> pd.Series:
    df = load_ohlcv(market, timeframe, bars=bars)
    if df.empty:
        return pd.Series(dtype=float)
    return df["close"].astype(float)


@st.cache_data(show_spinner=False)
def _build_returns(markets: tuple[str, ...], timeframe: str, use_log: bool,
                   bars: int) -> pd.DataFrame:
    closes = {}
    for m in markets:
        s = _load_close(m, timeframe, bars)
        if not s.empty:
            closes[m] = s
    if not closes:
        return pd.DataFrame()
    px = pd.DataFrame(closes).sort_index()
    px = px[~px.index.duplicated(keep="last")]
    px = px.where(px > 0)
    return np.log(px).diff() if use_log else px.pct_change()


def get_context() -> Context:
    """진입점(app.py)이 만들어 세션에 넣어둔 컨텍스트를 꺼낸다.

    사이드바 필터는 진입점에서 한 번만 그려지므로, 각 뷰는 결과만 읽어 쓴다.
    """
    ctx = st.session_state.get("_ctx")
    if ctx is None:  # 뷰를 단독 실행한 경우의 방어
        ctx = load_context()
    return ctx


def load_context() -> Context:
    """사이드바 필터를 그리고, 그 설정으로 계산까지 마친 컨텍스트를 돌려준다."""
    with st.sidebar:
        st.markdown("### 분석 설정")
        market = st.selectbox(
            "기준 코인", BASE_CANDIDATES, index=0, format_func=short, key="f_market",
            help="시장(market) 자리에 놓을 코인. 나머지 코인을 여기에 회귀시킵니다.",
        )
        timeframe = st.selectbox(
            "봉", ["1d", "1h"], index=0, key="f_timeframe",
            help=GLOSSARY["timeframe"],
        )
        period_label = st.selectbox(
            "기간", list(PERIODS.keys()), index=3, key="f_period",
        )
        ret_kind = st.radio(
            "수익률", ["로그", "단순"], index=0, horizontal=True, key="f_ret",
            help=GLOSSARY["log_return"],
        )
        common_span = st.toggle(
            "공통 구간", value=True, key="f_common",
            help=GLOSSARY["common_span"],
        )

    market_label = short(market)
    alts = [c for c in DEFAULT_COINS if c != market]
    use_log = ret_kind == "로그"
    bars_per_year = BARS_PER_YEAR[timeframe]

    # 고른 기간을 채우는 데 필요한 봉 수를 역산한다.
    # 로컬에 그만큼 있으면 API를 치지 않고, 없으면 받아와서 parquet으로 남긴다.
    req_days = PERIODS[period_label]
    if req_days is None:
        need_bars = UPBIT_MAX_FETCH_BARS
    else:
        need_bars = int(req_days * BARS_PER_DAY.get(timeframe, 1) * 1.05) + 50

    spinner_msg = (f"{timeframe} 데이터를 Upbit에서 받는 중… "
                   f"(처음 한 번만 걸립니다. 이후에는 로컬에 저장된 것을 씁니다)")
    with st.spinner(spinner_msg):
        rets_all = _build_returns(tuple(DEFAULT_COINS), timeframe, use_log, need_bars)

    if rets_all.empty or market not in rets_all.columns:
        if timeframe in BUNDLED_TIMEFRAMES:
            st.error(f"데이터를 찾을 수 없습니다. `data/ohlcv/` 를 확인하세요. ({OHLCV_DIR})")
        else:
            st.warning(
                f"**{timeframe} 데이터를 불러오지 못했습니다.**\n\n"
                "이 저장소에는 일봉(1d)만 동봉되어 있고, 나머지 타임프레임은 실행 시 "
                "Upbit API로 받아옵니다. Upbit이 해외 IP를 차단하는 경우가 있어 "
                "클라우드 환경에서는 실패할 수 있습니다."
            )
            st.info("사이드바의 **봉**에서 **1d**를 고르시면 동봉된 데이터로 바로 보실 수 있습니다.")
        st.stop()

    data_start, data_end = rets_all.index.min(), rets_all.index.max()
    days = PERIODS[period_label]
    if days is None:
        window_start = data_start
    else:
        # Include exactly the requested number of return bars.  The old
        # inclusive date window selected one extra observation at both 1d and 1h.
        bars_per_day = BARS_PER_DAY[timeframe]
        period_bars = days * bars_per_day
        window_start = data_end - pd.Timedelta(
            days=(period_bars - 1) / bars_per_day,
        )
    rets = rets_all[rets_all.index >= window_start]
    rets = rets.dropna(how="any") if common_span else rets.dropna(subset=[market])

    if len(rets) < 30:
        st.warning("표본이 30봉 미만입니다. 기간을 늘리거나 '공통 구간'을 끄세요.")
        st.stop()

    # 고른 기간을 실제로 채우지 못했으면 반드시 알린다.
    # 예전에는 "최근 1년"을 골라도 1,000봉(42일)만 분석되면서 화면에 아무 표시가 없었다.
    shortfall = None
    if req_days is not None:
        actual_days = (rets.index.max() - rets.index.min()).days
        if actual_days < req_days * 0.9:
            shortfall = (req_days, actual_days)

    results: dict[str, dict] = {}
    for m in alts:
        if m not in rets.columns:
            continue
        pair = rets[[market, m]].dropna()
        if len(pair) < 30:
            continue
        res = market_model(pair[m].values, pair[market].values, bars_per_year)
        if res:
            results[m] = res

    if not results:
        st.error("회귀 가능한 코인이 없습니다.")
        st.stop()

    table = pd.DataFrame(results).T
    table.index.name = "market"

    if shortfall:
        req_d, act_d = shortfall
        st.warning(
            f"**'{period_label}'을 골랐지만 실제로는 {act_d}일치만 분석됩니다.** "
            f"({rets.index.min().date()} ~ {rets.index.max().date()}, {len(rets):,}봉)\n\n"
            f"{timeframe} 데이터가 요청한 {req_d}일을 채우지 못했습니다. "
            "표본이 짧으면 우연히 유의한 결과가 나오기 쉬우니 숫자를 그대로 믿지 마세요."
        )

    # 사이드바 하단에 현재 슬라이스 요약
    with st.sidebar:
        st.markdown("---")
        st.caption(
            f"분석 구간 **{rets.index.min().date()} ~ {rets.index.max().date()}**  \n"
            f"{len(rets):,}봉 · 기준 {market_label}  \n"
            f"보유 데이터 {data_start.date()} ~ {data_end.date()}"
        )

    return Context(
        market=market, market_label=market_label, alts=alts,
        timeframe=timeframe, period_label=period_label,
        use_log=use_log, common_span=common_span, bars_per_year=bars_per_year,
        rets=rets, results=results, table=table,
        data_start=data_start, data_end=data_end,
        coins=list(results.keys()),
    )


def page_header(title: str, subtitle: str, ctx: Context) -> None:
    """페이지 제목 + 현재 설정 한 줄."""
    title_col, market_col = st.columns([4, 1])
    with title_col:
        st.title(title)
    with market_col:
        st.markdown(
            f'<div class="base-coin-badge"><span>\uae30\uc900</span><strong>{ctx.market_label}</strong></div>',
            unsafe_allow_html=True,
        )
    st.caption(
        f"{subtitle}  ·  {ctx.timeframe} · "
        f"{ctx.period_label} · {len(ctx.rets):,}봉"
    )
