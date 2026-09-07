"""데이터 로딩 — 동봉 parquet 우선, 없으면 Upbit API.

저장소에는 일봉(1d)만 동봉되어 있다. parquet이 21MB짜리 바이너리라
갱신할 때마다 git 히스토리에 통째로 쌓이기 때문이다.

시간봉(1h)처럼 동봉되지 않은 타임프레임은 실행 시 Upbit API로 받아온다.
단, Upbit이 해외 IP를 차단할 수 있어 클라우드 배포 환경에서는
실패할 수 있다. 그래서 실패를 예외로 던지지 않고 빈 값으로 돌려주고,
호출부가 안내 문구를 띄운 뒤 일봉으로 되돌아가게 한다.
"""

from __future__ import annotations

import time

import pandas as pd
import requests

from core.config import (
    OHLCV_DIR,
    UPBIT_MAX_FETCH_BARS,
    TIMEFRAMES,
    UPBIT_BASE_URL,
    UPBIT_MAX_COUNT,
    UPBIT_RATE_LIMIT_SEC,
)

_COLUMNS = ["open", "high", "low", "close", "volume", "value"]
_last_call = 0.0


def ohlcv_path(market: str, timeframe: str):
    return OHLCV_DIR / market / f"{timeframe}.parquet"


def load_local(market: str, timeframe: str) -> pd.DataFrame:
    """동봉된 parquet 읽기. 없으면 빈 DataFrame."""
    path = ohlcv_path(market, timeframe)
    if not path.exists():
        return pd.DataFrame(columns=_COLUMNS)

    df = pd.read_parquet(path).sort_index()
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    return df


# ----------------------------------------------------------------------
# Upbit
# ----------------------------------------------------------------------


def _rate_limit() -> None:
    global _last_call
    gap = time.time() - _last_call
    if gap < UPBIT_RATE_LIMIT_SEC:
        time.sleep(UPBIT_RATE_LIMIT_SEC - gap)
    _last_call = time.time()


def _candles_to_df(raw: list[dict]) -> pd.DataFrame:
    if not raw:
        return pd.DataFrame(columns=_COLUMNS)

    df = pd.DataFrame(raw)
    df["timestamp"] = pd.to_datetime(df["candle_date_time_utc"], utc=True)
    df = df.rename(
        columns={
            "opening_price": "open",
            "high_price": "high",
            "low_price": "low",
            "trade_price": "close",
            "candle_acc_trade_volume": "volume",
            "candle_acc_trade_price": "value",
        }
    )
    df = df.set_index("timestamp")[[c for c in _COLUMNS if c in df.columns]]
    return df.sort_index()[lambda d: ~d.index.duplicated(keep="last")]


def fetch_upbit(market: str, timeframe: str, bars: int = 1000) -> pd.DataFrame:
    """Upbit에서 최근 N봉을 받아온다. 실패하면 빈 DataFrame."""
    unit = TIMEFRAMES.get(timeframe)
    if unit is None:
        return pd.DataFrame(columns=_COLUMNS)

    url = f"{UPBIT_BASE_URL}/candles/{unit}"
    frames: list[pd.DataFrame] = []
    to = None
    remaining = bars

    try:
        while remaining > 0:
            _rate_limit()
            params = {"market": market, "count": min(UPBIT_MAX_COUNT, remaining)}
            if to is not None:
                params["to"] = to
            resp = requests.get(url, params=params, timeout=10)
            resp.raise_for_status()
            raw = resp.json()
            if not raw:
                break

            chunk = _candles_to_df(raw)
            if chunk.empty:
                break
            frames.append(chunk)
            remaining -= len(chunk)
            # 가장 오래된 캔들 시각을 다음 요청의 상한으로
            to = chunk.index.min().strftime("%Y-%m-%dT%H:%M:%SZ")
            if len(chunk) < min(UPBIT_MAX_COUNT, remaining + len(chunk)):
                break
    except (requests.RequestException, ValueError, KeyError):
        # 지역 차단·타임아웃·응답 형식 변경 — 호출부가 빈 값으로 처리
        return pd.DataFrame(columns=_COLUMNS)

    if not frames:
        return pd.DataFrame(columns=_COLUMNS)

    out = pd.concat(frames).sort_index()
    return out[~out.index.duplicated(keep="last")]


def save_local(market: str, timeframe: str, df: pd.DataFrame) -> None:
    """받아온 캔들을 parquet으로 남긴다.

    동봉하지 않는 타임프레임(1h 등)은 .gitignore 에 걸려 있어
    저장소 크기에 영향을 주지 않으면서 다음 실행부터 API를 치지 않게 해 준다.
    """
    path = ohlcv_path(market, timeframe)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        df.to_parquet(path)
    except Exception:
        pass  # 저장 실패는 조회를 막지 않는다


def load_ohlcv(market: str, timeframe: str, bars: int | None = None) -> pd.DataFrame:
    """로컬 parquet 우선. 부족하면 Upbit에서 채우고 저장한다.

    bars 는 "이만큼은 있어야 한다"는 요청량이다. 로컬이 그만큼 가지고 있으면
    API를 치지 않는다.
    """
    want = min(bars or 1000, UPBIT_MAX_FETCH_BARS)
    local = load_local(market, timeframe)
    if not local.empty and len(local) >= want:
        return local

    fetched = fetch_upbit(market, timeframe, bars=want)
    if fetched.empty:
        return local  # API 실패 — 있는 만큼이라도 돌려준다

    if not local.empty:
        fetched = pd.concat([local, fetched]).sort_index()
        fetched = fetched[~fetched.index.duplicated(keep="last")]

    if timeframe not in ("1d",):  # 동봉된 일봉은 덮어쓰지 않는다
        save_local(market, timeframe, fetched)
    return fetched
