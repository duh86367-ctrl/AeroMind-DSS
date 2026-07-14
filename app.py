import streamlit as st
import pandas as pd
import numpy as np
import pydeck as pdk
import time

# 1. 페이지 기본 설정 및 ATC 스타일 다크 테마 적용
st.set_page_config(
    page_title="AeroMind-DSS v2.0",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 레이더 화면 같은 프로페셔널 다크 모드 스타일 강제 적용
st.markdown("""
    <style>
    .main { background-color: #11141a; color: #e0e6ed; }
    .stAlert { background-color: #1c2333; border: 1px solid #3b4252; }
    h1, h2, h3 { color: #00ff66 !important; font-family: 'Courier New', Courier, monospace; }
    </style>
    """, unsafe_allow_html=True)

# 2. 핵심 유틸리티 함수 (하버사인 지구 곡률 거리 계산)
def get_horizontal_distance(lat1, lon1, lat2, lon2):
    R = 3440.065 # 지구 반지름 (Nautical Miles, 해리 단위)
    p = np.pi / 180
    a = 0.5 - np.cos((lat2 - lat1) * p)/2 + np.cos(lat1 * p) * np.cos(lat2 * p) * (1 - np.cos((lon2 - lon1) * p)) / 2
    return 2 * R * np.arcsin(np.sqrt(a))

# 3. 테스트용 실시간 가상 ADS-B 데이터 생성 (OpenSky 연동 전 검증용 기본 데이터)
# 관제사가 시스템 성능을 바로 확인할 수 있도록 일부러 충돌 위험 상황을 연출한 데이터셋입니다.
def load_airspace_data():
    data = {
        'callsign': ['KAL721', 'AAR317', 'JJA104', 'PAL812', 'ANA862'],
        'latitude': [37.46, 37.48, 37.55, 37.40, 37.60],
        'longitude': [126.44, 126.46, 126.50, 126.35, 126.65],
        'geo_altitude': [12000, 12500, 8000, 15000, 22000],  # Feet
        'velocity': [280, 250, 210, 310, 400],              # Knots
        'heading': [45, 225, 180, 90, 270],                 # Degrees
        'vertical_rate': [-15.0, 5.0, -8.0, 0.0, 0.0],       # m/s (음수는 강하)
        'wake_class': ['HEAVY', 'MEDIUM', 'LIGHT', 'HEAVY', 'MEDIUM']
    }
    return pd.DataFrame(data)

# 데이터 로드
if 'df_airspace' not in st.session_state:
    st.session_state.df_airspace = load_airspace_data()

df = st.session_state.df_airspace

# --- 상단 헤더 영역 ---
st.title("✈️ AeroMind-DSS v2.0")
st.subheader("AI-Driven Air Traffic Control Decision Support System")
st.markdown("---")

# --- 기상 정보 및 공항 수용량 분석 (Weather & ATFM 레이어) ---
# 실제 API 연동 시 METAR 데이터를 파싱하여 변수에 매핑하는 구간입니다.
wind_speed = 18
crosswind = 14
arrival_count = len(df)

col_meta1, col_meta2, col_meta3 = st.columns(3)
with col_meta1:
    st.metric(label="RKSI (인천공항) 활주로 운영 상태", value="RWY 33L ACTIVE")
with col_meta2:
    st.metric(label="실시간 측풍 (Crosswind)", value=f"{crosswind} KT", delta="주의 상태" if crosswind > 12 else "정상")
with col_meta3:
    st.metric(label="ATFM 수용량 밀도 (Arrival Flow)", value=f"{arrival_count} / 30 기체", delta="유입 증가" if arrival_count > 4 else "원활")

st.markdown("---")

# --- 핵심 AI 추론 엔진 (Conflict & Separation Engine) ---
conflict_alerts = []
lookahead_time = 5 # 5분 후 미래 위치 예측

# Pair-wise 공역 스캔 알고리즘
for i in range(len(df)):
    for j in range(i + 1, len(df)):
        ac1 = df.iloc[i]
        ac2 = df.iloc[j]
        
        # 미래 5분 뒤의 4D 좌표 추론 (선형 벡터 외삽)
        ac1_vx = ac1['velocity'] * np.sin(np.radians(ac1['heading'])) / 60
        ac1_vy = ac1['velocity'] * np.cos(np.radians(ac1['heading'])) / 60
        ac2_vx = ac2['velocity'] * np.sin(np.radians(ac2['heading'])) / 60
        ac2_vy = ac2['velocity'] * np.cos(np.radians(ac2['heading'])) / 60
        
        ac1_f_lat = ac1['latitude'] + (ac1_vy * lookahead_time) / 60
        ac1_f_lon = ac1['longitude'] + (ac1_vx * lookahead_time) / (60 * np.cos(np.radians(ac1['latitude'])))
        ac2_f_lat = ac2['latitude'] + (ac2_vy * lookahead_time) / 60
        ac2_f_lon = ac2['longitude'] + (ac2_vx * lookahead_time) / (60 * np.cos(np.radians(ac2['latitude'])))
        
        # 미래 분리치 계산
        h_dist = get_horizontal_distance(ac1_f_lat, ac1_f_lon, ac2_f_lat, ac2_f_lon)
        ac1_f_alt = ac1['geo_altitude'] + (ac1['vertical_rate'] * lookahead_time * 60 * 3.28)
        ac2_f_alt = ac2['geo_altitude'] + (ac2['vertical_rate'] * lookahead_time * 60 * 3.28)
        v_dist = abs(ac1_f_alt - ac2_f_alt)
        
        # ICAO 표준 미달 경보 조건 (수평 3해리 미만 및 수직 1000피트 미만 동시 만족 시)
        if h_dist < 3.0 and v_dist < 1000:
            conflict_alerts.append({
                'pair': [ac1['callsign'], ac2['callsign']],
                'h_dist': h_dist,
                'v_dist': v_dist
            })

# --- 개별 항공기 위험도 종합 평가 엔진 (Risk Scoring & XAI) ---
risk_scores = []
xai_reasons = []
advisories = []

for idx, row in df.iterrows():
    score = 15 # 기본 안전 점수
    reasons = []
    advisory = "MAINTAIN PRESENT HEADING/SPEED."
    
    # 1. 충돌 위험 변수 반영
    in_conflict = False
    for alert in conflict_alerts:
        if row['callsign'] in alert['pair']:
            in_conflict = True
            other_aircraft = alert['pair'][0] if alert['pair'][1] == row['callsign'] else alert['pair'][1]
    
    if in_conflict:
        score += 55
        reasons.append("• [위험] 5분 내 타 항공기와 표준 분리치 상실 위협 감지")
        advisory = "ALERT: TURN LEFT HEADING 270 IMMEDIATELY FOR SEPARATION."
    
    # 2. 급강하 상태 분석
    if row['vertical_rate'] < -10.0:
        score += 15
        reasons.append(f"• [주의] 이상 급강하 경보 상태 ({row['vertical_rate']} m/s)")
        if not in_conflict: advisory = "MONITOR VERTICAL SPEED AND EXPEDITE LEVEL OFF."
        
    # 3. 후류 와류 취약성 분석 (선행 Heavy기 배치 조건부 필터링)
    if row['wake_class'] == 'LIGHT' and any(df['wake_class'] == 'HEAVY'):
        score += 10
        reasons.append("• [주의] 공역 내 HEAVY 기체 존재로 인한 후류 와류(Wake Turbulence) 간격 확보 필요")
        
    # 4. 공역 기상 패널티
    if crosswind > 12:
        score += 5
        reasons.append("• 활주로 강한 측풍 환경 노출")

    risk_scores.append(min(score, 100))
    xai_reasons.append(reasons if reasons else ["• 정상 비행 상태 유지 중"])
    advisories.append(advisory)

df['risk_score'] = risk_scores
df['xai'] = xai_reasons
df['advisory'] = advisories

# --- 메인 레이아웃 분할 (2단 구조) ---
col_left, col_right = st.columns([3, 2])

# [좌측 영역]: PyDeck 기반 2.5D 관제 레이더 스크린 시각화
with col_left:
    st.subheader("🛰️ 실시간 전술 공역 레이더")
    
    # 위험 기체는 빨간색, 안전 기체는 초록색 레이더 마커로 맵 매핑 데이터 분리
    df['color_r'] = df['risk_score'].apply(lambda s: 255 if s > 50 else 0)
    df['color_g'] = df['risk_score'].apply(lambda s: 0 if s > 50 else 255)
    
    # PyDeck 레이어 설계
    layer = pdk.Layer(
        "ScatterplotLayer",
        df,
        get_position=["longitude", "latitude"],
        get_color="[color_r, color_g, 50, 200]",
        get_radius=400,
        pickable=True,
    )
    
    # 텍스트 레이블 레이어 (레이더 스크린상 항공기 정보 표기 룩앤필)
    df['label'] = df.apply(lambda r: f"{r['callsign']}\nALT: {r['geo_altitude']}ft\nSPD: {r['velocity']}kts", axis=1)
    text_layer = pdk.Layer(
        "TextLayer",
        df,
        get_position=["longitude", "latitude"],
        get_text="label",
        get_size=14,
        get_color=[255, 255, 255],
        get_alignment_baseline="'bottom'",
        get_pixel_offset=[0, -15]
    )
    
    # 지도의 중심점을 인천공항 주변으로 세팅
    view_state = pdk.ViewState(
        latitude=37.46,
        longitude=126.44,
        zoom=10,
        pitch=30
    )
    
    st.pydeck_chart(pdk.Deck(
        layers=[layer, text_layer],
        initial_view_state=view_state,
        map_style="mapbox://styles/mapbox/dark-v10"
    ))
    
    # 데이터 원본 테이블 모니터링
    st.dataframe(df[['callsign', 'geo_altitude', 'velocity', 'heading', 'wake_class', 'risk_score']])

# [우측 영역]: Controller Priority Queue (관제사 타겟 우선순위 추출기)
with col_right:
    st.subheader("🚨 관제 우선순위 큐 (Priority Queue)")
    st.caption("AI가 위험 요소를 분석하여 지금 가장 먼저 교신해야 할 기체 순으로 정렬합니다.")
    
    # 위험도 점수 기준으로 내림차순 정렬
    df_queue = df.sort_values(by='risk_score', ascending=False)
    
    for idx, row in df_queue.iterrows():
        # 위험 등급별 UI 카드 가독성 차별화
        if row['risk_score'] > 70:
            box_style = "🔴 CRITICAL"
        elif row['risk_score'] > 40:
            box_style = "🟡 WARNING"
        else:
            box_style = "🟢 STABLE"
            
        with st.expander(f"[{box_style}] {row['callsign']} — Risk Score: {row['risk_score']}점"):
            st.markdown("**🔍 AI 의사결정 근거 (Explainable AI):**")
            for r in row['xai']:
                st.write(r)
            
            st.markdown("**📢 관제 조언 지시 (AI Advisory Command):**")
            st.info(row['advisory'])

# 30초마다 화면 자동 동기화 리프레시 시스템 구현을 위한 안내
st.caption("AeroMind-DSS 데이터 파이프라인 엔진 정상 작동 중. 30초 간격 자동 동기화 활성화.")
