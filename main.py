import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from scipy.stats import ttest_ind
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, auc
from sklearn.preprocessing import LabelEncoder
import io

# -----------------------------------------------------------------------------
# 1. 기본 설정 및 폰트
# -----------------------------------------------------------------------------
st.set_page_config(page_title="데이터 분석 및 모델링 도구", layout="wide")

# 한글 폰트 설정 (Mac/Window 환경 대응)
import platform
system_name = platform.system()
if system_name == 'Darwin': # Mac
    plt.rc('font', family='AppleGothic') 
elif system_name == 'Windows': # Windows
    plt.rc('font', family='Malgun Gothic')
else: # Linux (Colab 등)
    plt.rc('font', family='NanumGothic')
plt.rc('axes', unicode_minus=False)

# 세션 상태 초기화 (페이지 간 데이터 공유를 위해)
if 'df' not in st.session_state:
    st.session_state['df'] = None
if 'df_processed' not in st.session_state:
    st.session_state['df_processed'] = None
if 'target_col' not in st.session_state:
    st.session_state['target_col'] = None
if 'selected_features' not in st.session_state:
    st.session_state['selected_features'] = None

# -----------------------------------------------------------------------------
# 2. 사이드바 메뉴 구성
# -----------------------------------------------------------------------------
st.sidebar.title("분석 단계 (Menu)")
menu = ["1. 데이터 업로드", "2. 데이터 탐색 및 시각화", "3. 데이터 전처리 (T-test)", "4. 모형 구축 및 평가"]
choice = st.sidebar.radio("단계를 선택하세요", menu)

# -----------------------------------------------------------------------------
# [페이지 1] 데이터 업로드
# -----------------------------------------------------------------------------
if choice == "1. 데이터 업로드":
    st.title("📂 데이터 업로드")
    
    uploaded_file = st.file_uploader("CSV 또는 Excel 파일을 업로드하세요", type=['csv', 'xlsx'])
    
    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
            
            st.session_state['df'] = df
            st.success(f"데이터 로드 성공! (행: {df.shape[0]}, 열: {df.shape[1]})")
            st.dataframe(df.head())
            
        except Exception as e:
            st.error(f"데이터를 읽는 중 오류가 발생했습니다: {e}")
            
    elif st.session_state['df'] is not None:
        st.info("이미 업로드된 데이터가 있습니다.")
        st.dataframe(st.session_state['df'].head())

# -----------------------------------------------------------------------------
# [페이지 2] 데이터 탐색 및 시각화
# -----------------------------------------------------------------------------
elif choice == "2. 데이터 탐색 및 시각화":
    st.title("🔍 데이터 탐색 및 시각화")
    
    if st.session_state['df'] is None:
        st.warning("먼저 '데이터 업로드' 메뉴에서 데이터를 업로드해주세요.")
    else:
        df = st.session_state['df']
        
        st.subheader("1. 데이터 기본 정보")
        buffer = io.StringIO()
        df.info(buf=buffer)
        s = buffer.getvalue()
        st.text(s)
        
        st.subheader("2. 기술 통계량")
        st.dataframe(df.describe())
        
        st.subheader("3. 상관관계 히트맵 (수치형 변수)")
        numeric_df = df.select_dtypes(include=['float64', 'int64'])
        if not numeric_df.empty:
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(numeric_df.corr(), annot=True, fmt=".2f", cmap='coolwarm', ax=ax)
            st.pyplot(fig)
        else:
            st.info("수치형 변수가 없어 히트맵을 그릴 수 없습니다.")

# -----------------------------------------------------------------------------
# [페이지 3] 데이터 전처리 (T-test 기반 변수 선택)
# -----------------------------------------------------------------------------
elif choice == "3. 데이터 전처리 (T-test)":
    st.title("⚙️ 데이터 전처리 및 변수 선택")
    
    if st.session_state['df'] is None:
        st.warning("먼저 '데이터 업로드' 메뉴에서 데이터를 업로드해주세요.")
    else:
        df = st.session_state['df'].copy()
        
        # 1. 결측치 처리
        st.subheader("1. 결측치 처리")
        missing_method = st.radio("결측치 처리 방식", ["삭제 (Drop)", "평균 대치 (Mean)", "0으로 채움"], horizontal=True)
        
        if missing_method == "삭제 (Drop)":
            df = df.dropna()
        elif missing_method == "평균 대치 (Mean)":
            numeric_cols = df.select_dtypes(include=['number']).columns
            df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
        else:
            df = df.fillna(0)
            
        st.write("결측치 처리 완료. 현재 데이터 크기:", df.shape)

        # 2. 타겟 변수 설정
        st.subheader("2. 타겟 변수(Y) 선택")
        target = st.selectbox("분석할 타겟 변수(이진 분류 권장)를 선택하세요", df.columns)
        st.session_state['target_col'] = target
        
        # 타겟 변수 인코딩 (문자열일 경우 숫자로 변환)
        if df[target].dtype == 'object':
            le = LabelEncoder()
            df[target] = le.fit_transform(df[target])
            st.info(f"타겟 변수 '{target}'가 수치형으로 인코딩되었습니다.")

        # 3. T-test를 이용한 변수 선택 (P-value <= 0.05)
        st.subheader("3. T-test 기반 변수 선택")
        st.markdown("**기준:** P-value가 0.05 이하인 변수만 선택합니다.")
        
        # 타겟 클래스 확인 (이진 분류여야 T-test 적합)
        unique_targets = df[target].unique()
        
        if len(unique_targets) == 2:
            group0 = df[df[target] == unique_targets[0]]
            group1 = df[df[target] == unique_targets[1]]
            
            numeric_features = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
            if target in numeric_features:
                numeric_features.remove(target)
            
            selected_features = []
            t_test_results = []
            
            for col in numeric_features:
                # 각 그룹의 해당 컬럼 데이터 추출
                val0 = group0[col]
                val1 = group1[col]
                
                # T-test 수행
                t_stat, p_val = ttest_ind(val0, val1, equal_var=False) # 이분산 가정
                
                # 결과 저장
                is_significant = p_val <= 0.05
                t_test_results.append({
                    "Feature": col,
                    "P-value": round(p_val, 5),
                    "Significant": is_significant
                })
                
                # P-value 0.05 이하만 선택
                if is_significant:
                    selected_features.append(col)
            
            # 결과 테이블 표시
            results_df = pd.DataFrame(t_test_results)
            st.write("T-test 결과 요약:")
            st.dataframe(results_df)
            
            if selected_features:
                st.success(f"P-value <= 0.05 조건을 만족하는 변수 {len(selected_features)}개를 선택했습니다.")
                st.write(f"선택된 변수: {selected_features}")
                
                # 선택된 변수와 타겟만 포함하여 저장
                final_cols = selected_features + [target]
                st.session_state['df_processed'] = df[final_cols]
                st.session_state['selected_features'] = selected_features
            else:
                st.error("조건을 만족하는 변수가 하나도 없습니다. 기준을 완화하거나 데이터를 확인하세요.")
                st.session_state['df_processed'] = None
                
        else:
            st.error(f"선택한 타겟 변수의 클래스가 {len(unique_targets)}개입니다. T-test는 2개의 그룹(이진 분류)일 때 가장 적합합니다.")

# -----------------------------------------------------------------------------
# [페이지 4] 모형 구축 및 평가
# -----------------------------------------------------------------------------
elif choice == "4. 모형 구축 및 평가":
    st.title("🤖 모형 구축 및 평가")
    
    if st.session_state['df_processed'] is None:
        st.warning("먼저 '3. 데이터 전처리' 단계에서 변수 선택을 완료해주세요.")
    else:
        df_final = st.session_state['df_processed']
        target = st.session_state['target_col']
        features = st.session_state['selected_features']
        
        st.write(f"학습에 사용할 변수: {features}")
        
        # X, y 분리
        X = df_final[features]
        y = df_final[target]
        
        # 학습/테스트 데이터 분리
        test_size = st.slider("테스트 데이터 비율 설정", 0.1, 0.5, 0.2)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
        
        # 모델 학습 (로지스틱 회귀 예시)
        if st.button("모델 학습 시작"):
            model = LogisticRegression()
            model.fit(X_train, y_train)
            
            # 예측
            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test)[:, 1]
            
            # 평가
            st.subheader("1. 성능 평가 지표")
            acc = accuracy_score(y_test, y_pred)
            st.metric("Accuracy (정확도)", f"{acc:.4f}")
            
            st.text("Classification Report:")
            st.text(classification_report(y_test, y_pred))
            
            # 혼동 행렬
            st.subheader("2. 혼동 행렬 (Confusion Matrix)")
            cm = confusion_matrix(y_test, y_pred)
            fig_cm, ax_cm = plt.subplots()
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax_cm)
            ax_cm.set_xlabel('Predicted')
            ax_cm.set_ylabel('Actual')
            st.pyplot(fig_cm)
            
            # ROC 커브
            st.subheader("3. ROC Curve")
            fpr, tpr, _ = roc_curve(y_test, y_proba)
            roc_auc = auc(fpr, tpr)
            
            fig_roc, ax_roc = plt.subplots()
            ax_roc.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
            ax_roc.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            ax_roc.set_xlim([0.0, 1.0])
            ax_roc.set_ylim([0.0, 1.05])
            ax_roc.set_xlabel('False Positive Rate')
            ax_roc.set_ylabel('True Positive Rate')
            ax_roc.set_title('Receiver Operating Characteristic')
            ax_roc.legend(loc="lower right")
            st.pyplot(fig_roc)
