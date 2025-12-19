import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, confusion_matrix, roc_curve, auc, classification_report)

# --------------------------------------------------------------------------------
# 1. 기본 설정 및 세션 초기화
# --------------------------------------------------------------------------------
st.set_page_config(page_title="Logistic Regression App", layout="wide")

# 한글 폰트 설정 (필요시 운영체제에 맞게 주석 해제하여 사용)
plt.rc('font', family='Malgun Gothic') # Windows 예시
plt.rc('axes', unicode_minus=False)

# 세션 상태 초기화
if 'df' not in st.session_state:
    st.session_state['df'] = None
if 'df_processed' not in st.session_state:
    st.session_state['df_processed'] = None
if 'target_col' not in st.session_state:
    st.session_state['target_col'] = None
if 'selected_features' not in st.session_state:
    st.session_state['selected_features'] = []

# 사이드바 메뉴
st.sidebar.title("분석 단계")
menu = ["1. 데이터 업로드", "2. 데이터 탐색 및 시각화", "3. 데이터 전처리 (T-test)", "4. 모형 구축 및 평가"]
choice = st.sidebar.radio("메뉴를 선택하세요", menu)

# --------------------------------------------------------------------------------
# [PAGE 1] 데이터 업로드
# --------------------------------------------------------------------------------
if choice == "1. 데이터 업로드":
    st.title("📂 데이터 업로드")
    
    uploaded_file = st.file_uploader("CSV 파일을 업로드하세요", type=["csv"])
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.session_state['df'] = df
            st.success("데이터 로드 성공!")
            st.write(f"데이터 크기: {df.shape[0]} 행, {df.shape[1]} 열")
            st.dataframe(df.head())
        except Exception as e:
            st.error(f"파일을 읽는 중 오류가 발생했습니다: {e}")
            
    elif st.session_state['df'] is not None:
        st.info("이미 업로드된 데이터가 있습니다.")
        st.dataframe(st.session_state['df'].head())

# --------------------------------------------------------------------------------
# [PAGE 2] 데이터 탐색 및 시각화
# --------------------------------------------------------------------------------
elif choice == "2. 데이터 탐색 및 시각화":
    st.title("🔍 데이터 탐색 및 시각화")
    
    if st.session_state['df'] is None:
        st.warning("먼저 '1. 데이터 업로드' 메뉴에서 데이터를 업로드해주세요.")
    else:
        df = st.session_state['df']
        
        st.subheader("1. 기술 통계량")
        st.dataframe(df.describe())
        
        st.subheader("2. 그래프 시각화")
        col1, col2 = st.columns([1, 3])
        
        with col1:
            viz_type = st.selectbox("그래프 유형", ["Histogram", "Box Plot", "Scatter Plot", "Bar Chart"])
            x_col = st.selectbox("X축 변수", df.columns)
            y_col = st.selectbox("Y축 변수 (선택)", [None] + list(df.columns))
            
        with col2:
            fig, ax = plt.subplots(figsize=(10, 5))
            try:
                if viz_type == "Histogram":
                    sns.histplot(data=df, x=x_col, kde=True, ax=ax)
                elif viz_type == "Box Plot":
                    sns.boxplot(data=df, x=x_col, y=y_col, ax=ax)
                elif viz_type == "Scatter Plot":
                    if y_col: sns.scatterplot(data=df, x=x_col, y=y_col, ax=ax)
                    else: st.warning("Scatter Plot은 Y축 변수가 필요합니다.")
                elif viz_type == "Bar Chart":
                    if y_col: sns.barplot(data=df, x=x_col, y=y_col, ax=ax)
                    else: df[x_col].value_counts().plot(kind='bar', ax=ax)
                
                st.pyplot(fig)
            except Exception as e:
                st.error(f"그래프 오류: {e}")

# --------------------------------------------------------------------------------
# [PAGE 3] 데이터 전처리 (T-test 변수 선택)
# --------------------------------------------------------------------------------
elif choice == "3. 데이터 전처리 (T-test)":
    st.title("⚙️ 데이터 전처리 및 변수 선택")
    
    if st.session_state['df'] is None:
        st.warning("먼저 데이터를 업로드해주세요.")
    else:
        df = st.session_state['df'].copy()
        
        # 1. 결측치 처리
        st.subheader("1. 결측치 처리")
        # 간단하게 숫자형은 평균, 범주형은 최빈값으로 채움
        num_cols = df.select_dtypes(include=['number']).columns
        cat_cols = df.select_dtypes(include=['object']).columns
        
        if len(num_cols) > 0:
            imputer_num = SimpleImputer(strategy='mean')
            df[num_cols] = imputer_num.fit_transform(df[num_cols])
        if len(cat_cols) > 0:
            imputer_cat = SimpleImputer(strategy='most_frequent')
            df[cat_cols] = imputer_cat.fit_transform(df[cat_cols])
            
        st.write("결측치 처리가 완료되었습니다.")

        # 2. 타겟 변수 선택
        st.subheader("2. 타겟 변수(Y) 설정")
        target_col = st.selectbox("분석할 타겟 변수(Target)를 선택하세요", df.columns)
        st.session_state['target_col'] = target_col
        
        # 타겟 인코딩 (문자열일 경우 숫자로 변환)
        if df[target_col].dtype == 'object':
            le = LabelEncoder()
            df[target_col] = le.fit_transform(df[target_col])
            st.info(f"타겟 변수 '{target_col}'가 수치형으로 인코딩되었습니다.")

        # 3. T-test 기반 변수 선택
        st.subheader("3. T-test 기반 변수 선택 (P-value <= 0.05)")
        
        if st.button("T-test 변수 선택 실행"):
            # 타겟 클래스 확인 (이진 분류 가정)
            unique_targets = df[target_col].unique()
            
            if len(unique_targets) == 2:
                group0 = df[df[target_col] == unique_targets[0]]
                group1 = df[df[target_col] == unique_targets[1]]
                
                # 수치형 변수만 추출 (타겟 제외)
                candidate_features = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
                if target_col in candidate_features:
                    candidate_features.remove(target_col)
                
                selected_features = []
                results = []
                
                for col in candidate_features:
                    val0 = group0[col]
                    val1 = group1[col]
                    
                    # T-test 수행 (이분산 가정 equal_var=False)
                    t_stat, p_val = stats.ttest_ind(val0, val1, equal_var=False)
                    
                    is_select = p_val <= 0.05
                    results.append({'Variable': col, 'P-value': p_val, 'Selected': is_select})
                    
                    if is_select:
                        selected_features.append(col)
                
                # 결과 출력
                res_df = pd.DataFrame(results)
                st.dataframe(res_df.style.applymap(lambda x: 'background-color: lightgreen' if x is True else '', subset=['Selected']))
                
                if len(selected_features) > 0:
                    st.success(f"P-value 0.05 이하인 변수 {len(selected_features)}개가 선택되었습니다.")
                    st.write(f"**선택된 변수 목록:** {selected_features}")
                    
                    # 선택된 변수 + 타겟만 저장
                    st.session_state['selected_features'] = selected_features
                    st.session_state['df_processed'] = df[selected_features + [target_col]]
                else:
                    st.error("조건을 만족하는 변수가 하나도 없습니다. 데이터를 확인하세요.")
                    st.session_state['df_processed'] = None
            else:
                st.error("타겟 변수의 클래스가 2개가 아닙니다. (이진 분류 문제에서만 T-test 적용 가능)")

# --------------------------------------------------------------------------------
# [PAGE 4] 모형 구축 및 평가
# --------------------------------------------------------------------------------
elif choice == "4. 모형 구축 및 평가":
    st.title("🤖 모형 구축 및 평가")
    
    if st.session_state['df_processed'] is None:
        st.warning("먼저 '3. 데이터 전처리' 단계에서 변수 선택을 완료해주세요.")
    else:
        df_final = st.session_state['df_processed']
        target_col = st.session_state['target_col']
        features = st.session_state['selected_features']
        
        st.write(f"**학습에 사용되는 변수 ({len(features)}개):** {features}")
        
        # X, y 분리
        X = df_final[features]
        y = df_final[target_col]
        
        # Train/Test 분리
        test_size = st.slider("테스트 데이터 비율", 0.1, 0.5, 0.2)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
        
        # 스케일링 (로지스틱 회귀 성능 향상)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        if st.button("모델 학습 시작"):
            # 로지스틱 회귀 모델 학습
            model = LogisticRegression(max_iter=1000)
            model.fit(X_train_scaled, y_train)
            
            y_pred = model.predict(X_test_scaled)
            y_proba = model.predict_proba(X_test_scaled)[:, 1]
            
            # 평가 지표
            st.subheader("1. 성능 평가 지표")
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Accuracy", f"{accuracy_score(y_test, y_pred):.4f}")
            col2.metric("Precision", f"{precision_score(y_test, y_pred, zero_division=0):.4f}")
            col3.metric("Recall", f"{recall_score(y_test, y_pred, zero_division=0):.4f}")
            col4.metric("F1 Score", f"{f1_score(y_test, y_pred, zero_division=0):.4f}")
            
            # 혼동 행렬
            st.subheader("2. 혼동 행렬 (Confusion Matrix)")
            cm = confusion_matrix(y_test, y_pred)
            fig_cm, ax_cm = plt.subplots(figsize=(6, 4))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax_cm)
            ax_cm.set_xlabel('Predicted')
            ax_cm.set_ylabel('Actual')
            st.pyplot(fig_cm)
            
            # ROC 커브
            st.subheader("3. ROC Curve")
            if len(y.unique()) == 2:
                fpr, tpr, _ = roc_curve(y_test, y_proba)
                roc_auc = auc(fpr, tpr)
                
                fig_roc, ax_roc = plt.subplots(figsize=(6, 4))
                ax_roc.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
                ax_roc.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
                ax_roc.set_xlim([0.0, 1.0])
                ax_roc.set_ylim([0.0, 1.05])
                ax_roc.set_xlabel('False Positive Rate')
                ax_roc.set_ylabel('True Positive Rate')
                ax_roc.set_title('Receiver Operating Characteristic')
                ax_roc.legend(loc="lower right")
                st.pyplot(fig_roc)
            else:
                st.info("이진 분류가 아니어서 ROC Curve를 그릴 수 없습니다.")
