"""코인 영향력 분석 — 진입점.

시장(market) 자리에 기준 코인을 놓고 다른 코인의 수익률을 회귀시키는 시장모형:

    r_i,t = α + β · r_m,t + ε_t

- β   : 기준 코인이 1% 움직일 때 이 코인이 몇 % 움직이나 (민감도 · 증폭률)
- R²  : 이 코인 움직임 중 기준 코인으로 설명되는 비율 → "영향력"의 직접적인 답
- α   : 기준 코인으로 설명되지 않는 고유 수익 (연율화)
- σ_ε : 기준 코인 영향을 제거하고 남은 이 코인만의 변동성 (연율화)

화면은 답하는 질문별로 나눠 두고, 같이 봐야 의미가 생기는 것만 한 페이지에 묶었다.
사이드바 필터는 여기서 한 번만 그려 모든 페이지에 공유한다.

실행: streamlit run app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import streamlit as st

from core.theme import CARD_CSS
from core.ui import load_context

st.set_page_config(page_title="코인 영향력", page_icon="🧲", layout="wide")
st.markdown(CARD_CSS, unsafe_allow_html=True)

# ----------------------------------------------------------------------
# 카테고리 — 답하는 질문으로 묶는다
# ----------------------------------------------------------------------

nav = st.navigation(
    {
        "시작": [
            st.Page("views/intro.py", title="홈", icon="🧲", default=True),
        ],
        "얼마나 따라가나": [
            st.Page("views/rank.py", title="전체 순위", icon="📊"),
            st.Page("views/detail.py", title="코인 상세", icon="🔍"),
        ],
        "그 관계를 믿어도 되나": [
            st.Page("views/stability.py", title="안정성", icon="📈"),
            st.Page("views/leadlag.py", title="예측력", icon="⏱"),
        ],
        "포트폴리오": [
            st.Page("views/correlation.py", title="상관관계", icon="🔗"),
        ],
    }
)

# 사이드바 필터를 한 번만 그리고, 계산 결과를 세션에 담아 각 뷰가 꺼내 쓰게 한다.
st.session_state["_ctx"] = load_context()

nav.run()
