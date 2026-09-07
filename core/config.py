"""앱 설정 — 코인 목록, 타임프레임, 경로.

외부 의존성 없이 이 파일만으로 앱이 돌아간다.
"""

from __future__ import annotations

from pathlib import Path

# ======================================================================
# 경로
# ======================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
OHLCV_DIR = DATA_DIR / "ohlcv"

# ======================================================================
# 코인 (Upbit KRW 마켓)
# ======================================================================

DEFAULT_COINS: list[str] = [
    "KRW-BTC",
    "KRW-ETH",
    "KRW-XRP",
    "KRW-SOL",
    "KRW-DOGE",
    "KRW-DOT",
    "KRW-NEAR",
    "KRW-SUI",
    "KRW-TRX",
]

# 기준 자산으로 고를 수 있는 코인
BASE_CANDIDATES: list[str] = ["KRW-BTC", "KRW-ETH", "KRW-SOL", "KRW-XRP"]

# ======================================================================
# 타임프레임
# ======================================================================

# Upbit REST API 경로 매핑
TIMEFRAMES: dict[str, str] = {
    "1d": "days",
    "1h": "minutes/60",
}

# 저장소에 동봉된 타임프레임. 나머지는 실행 시 Upbit API로 받아온다.
BUNDLED_TIMEFRAMES: list[str] = ["1d"]

# 연율화 계수 — 암호화폐는 24/7이므로 365일 기준
BARS_PER_YEAR: dict[str, float] = {
    "1d": 365,
    "1h": 24 * 365,
}

# ======================================================================
# Upbit API
# ======================================================================

UPBIT_BASE_URL = "https://api.upbit.com/v1"
UPBIT_MAX_COUNT = 200          # 요청당 최대 캔들 수
UPBIT_RATE_LIMIT_SEC = 0.15    # 요청 간 최소 간격

# 한 번에 받아올 최대 봉 수.
# 요청당 200봉이라 8,760봉이면 코인 하나에 44회. 9개 코인이면 396회다.
# 그래서 받은 것은 parquet으로 저장해 두 번째부터는 API를 치지 않는다
# (동봉하지 않는 타임프레임이므로 .gitignore 에 걸려 저장소는 커지지 않는다).
UPBIT_MAX_FETCH_BARS = 20_000
