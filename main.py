import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, auc, ConfusionMatrixDisplay
)
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import SequentialFeatureSelector

# 페이지 기본 설정
st.set_page_config(page_title="로짓 모형 분석기", layout="wide")

st.title("📊 Logistic Regression Modeling Tool")
st.markdown("---")

# 세션 상태 초기화
if 'df' not in st.session_state:
    st.session_state['df'] = None

# ==========================================
# 1. 데이터 업로드
# ==========================================
st.header("1. 데이터 업로드")
uploaded_file = st.file_uploader("CSV 파일을 업로드하세요", type=["csv"])

if uploaded_file is not None:
    st.session_state['df'] = pd.read_csv(uploaded_file)
    st.success("데이터 업로드 성공!")
    st.dataframe(st.session_state['df'].head())

# 데이터가 로드된 경우에만 실행
if st.session_state['df'] is not None:
    df = st.session_state['df']

    st.markdown("---")
    # ==========================================
    # 2. 데이터 탐색 및 시각화
    # ==========================================
    st.header("2. 데이터 탐색 및 시각화 (EDA)")

    # ------------------------------------------
    # 2-1. T-test (P-value <= 0.05 자동 필터링)
    # ------------------------------------------
    st.subheader("가설 검정 (Significant Variables T-test)")
    st.caption("그룹 변수를 선택하면, 나머지 모든 수치형 변수에 대해 T-test를 수행하여 P-value가 0.05 이하인 변수만 보여줍니다.")

    # 그룹 변수 선택 (이진 분류 기준)
    group_col = st.selectbox("그룹 변수 (이진 범주형) 선택", df.columns, key='ttest_group_auto')

    if st.button("유의한 변수 찾기 (T-test 실행)"):
        # 그룹 변수 유효성 검사
        if df[group_col].nunique() != 2:
            st.error(f"오류: 선택한 그룹 변수 '{group_col}'의 고유값은 정확히 2개여야 합니다. (현재: {df[group_col].nunique()}개)")
        else:
            # 수치형 변수만 추출 (그룹 변수 자체는 제외)
            numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
            if group_col in numeric_cols:
                numeric_cols.remove(group_col)
            
            significant_vars = []
            groups = df[group_col].unique()
            
            # 모든 수치형 변수에 대해 반복 검정
            for col in numeric_cols:
                try:
                    group_a = df[df[group_col] == groups[0]][col]
                    group_b = df[df[group_col] == groups[1]][col]
                    
                    # 결측치 제외 후 t-test
                    t_stat, p_val = stats.ttest_ind(group_a, group_b, equal_var=False, nan_policy='omit')
                    
                    # P-value 0.05 이하인 경우만 저장
                    if p_val <= 0.05:
                        significant_vars.append({
                            "변수명": col,
                            "T-statistic": round(t_stat, 4),
                            "P-value": round(p_val, 5) # 소수점 5자리까지 표시
                        })
                except Exception as e:
                    pass # 계산 불가한 컬럼은 패스

            # 결과 출력
            if len(significant_vars) > 0:
                st.success(f"P-value <= 0.05인 유의미한 변수 {len(significant_vars)}개를 찾았습니다.")
                st.dataframe(pd.DataFrame(significant_vars))
            else:
                st.warning("P-value가 0.05 이하인 변수가 하나도 없습니다.")

    st.markdown("---")

    # ------------------------------------------
    # 2-2. 그래프 시각화
    # ------------------------------------------
    st.subheader("그래프 시각화")
    v_col1, v_col2, v_col3 = st.columns(3)
    
    with v_col1:
        x_axis = st.selectbox("X축 (X Label) 선택", df.columns)
    with v_col2:
        y_axis = st.selectbox("Y축 (Y Label) 선택 (선택 사항)", [None] + list(df.columns))
    with v_col3:
        plot_type = st.selectbox("그래프 유형", 
                                 ["Histogram", "Box Plot", "Scatter Plot", "Bar Chart", "Line Chart"])

    if st.button("그래프 그리기"):
        fig, ax = plt.subplots(figsize=(10, 5))
        try:
            if plot_type == "Histogram":
                sns.histplot(data=df, x=x_axis, kde=True, ax=ax)
            elif plot_type == "Box Plot":
                sns.boxplot(data=df, x=x_axis, y=y_axis, ax=ax)
            elif plot_type == "Scatter Plot":
                if y_axis:
                    sns.scatterplot(data=df, x=x_axis, y=y_axis, ax=ax)
                else:
                    st.warning("Scatter Plot은 Y축 선택이 필수입니다.")
            elif plot_type == "Bar Chart":
                if y_axis:
                    sns.barplot(data=df, x=x_axis, y=y_axis, ax=ax)
                else:
                    st.countplot(data=df, x=x_axis, ax=ax)
            elif plot_type == "Line Chart":
                if y_axis:
                    sns.lineplot(data=df, x=x_axis, y=y_axis, ax=ax)
                else:
                    st.warning("Line Chart는 Y축 선택이 필수입니다.")
            
            st.pyplot(fig)
        except Exception as e:
            st.error(f"그래프 오류: {e}")

    st.markdown("---")
    # ==========================================
    # 3. 데이터 전처리
    # ==========================================
    st.header("3. 데이터 전처리")
    
    col_p1, col_p2, col_p3 = st.columns(3)
    with col_p1:
        handle_na = st.checkbox("결측치 제거 (Drop NA)")
    with col_p2:
        do_scaling = st.checkbox("특성 스케일링 (StandardScaler)")
    with col_p3:
        do_encoding = st.checkbox("원-핫 인코딩 (범주형 변수)")

    if st.button("데이터 전처리 적용"):
        df_processed = df.copy()
        
        # 1. 결측치 처리
        if handle_na:
            df_processed = df_processed.dropna()
            st.info(f"결측치 제거 완료 (남은 행: {len(df_processed)})")

        # 2. 원-핫 인코딩
        if do_encoding:
            cat_cols = df_processed.select_dtypes(include=['object', 'category']).columns
            if len(cat_cols) > 0:
                df_processed = pd.get_dummies(df_processed, columns=cat_cols, drop_first=True)
                st.info("원-핫 인코딩 완료")
        
        # 3. 스케일링
        if do_scaling:
            num_cols = df_processed.select_dtypes(include=np.number).columns
            scaler = StandardScaler()
            df_processed[num_cols] = scaler.fit_transform(df_processed[num_cols])
            st.info("스케일링 완료")

        st.session_state['df_processed'] = df_processed
        st.success("전처리된 데이터가 저장되었습니다.")
        st.write(st.session_state['df_processed'].head())

    current_df = st.session_state.get('df_processed', df)

    st.markdown("---")
    # ==========================================
    # 4. 특성 선택 (Stepwise)
    # ==========================================
    st.header("4. 특성 선택 (Stepwise Selection)")

    target_col = st.selectbox("종속 변수 (Target) 선택", current_df.columns)
    feature_candidates = [c for c in current_df.columns if c != target_col]
    selected_features_pool = st.multiselect("변수 후보군 선택 (Stepwise에 사용할 변수들)", feature_candidates, default=feature_candidates)

    if st.button("전진 선택법(Stepwise) 실행"):
        if not selected_features_pool:
            st.warning("후보 변수를 선택해주세요.")
        else:
            X_temp = current_df[selected_features_pool]
            y_temp = current_df[target_col]
            
            # y 인코딩 체크
            if y_temp.dtype == 'object':
                le = LabelEncoder()
                y_temp = le.fit_transform(y_temp)

            try:
                # 데이터 절반 정도를 뽑도록 설정 ('auto')
                model_sel = LogisticRegression(solver='liblinear')
                sfs = SequentialFeatureSelector(model_sel, direction='forward', n_features_to_select='auto')
                
                with st.spinner("최적 변수 탐색 중..."):
                    sfs.fit(X_temp, y_temp)
                
                selected_mask = sfs.get_support()
                recommended_features = np.array(selected_features_pool)[selected_mask]
                
                st.success(f"Stepwise 결과 추천 변수: {', '.join(recommended_features)}")
                st.info("아래 '최종 변수 선택' 단계에서 이 변수들을 참고하여 선택하세요.")
            
            except Exception as e:
                st.error(f"Stepwise 오류: {e}")

    st.markdown("---")
    # ==========================================
    # 5. 데이터 나누기 & 최종 변수 선택
    # ==========================================
    st.header("5. 데이터 나누기 및 최종 변수 선택")

    col_split1, col_split2 = st.columns(2)
    with col_split1:
        final_features = st.multiselect("최종 독립 변수 (X) 선택", feature_candidates)
    with col_split2:
        test_size = st.slider("테스트 데이터 비율 (Test Size)", 0.1, 0.5, 0.2)

    st.markdown("---")
    # ==========================================
    # 6. 모형 구축 및 평가
    # ==========================================
    st.header("6. 모형 구축 및 평가")

    if st.button("모델 학습 및 평가"):
        if not final_features:
            st.error("독립 변수를 선택해야 합니다.")
        else:
            X = current_df[final_features]
            y = current_df[target_col]

            # y 전처리 (문자열일 경우 숫자로)
            if y.dtype == 'object' or y.dtype.name == 'category':
                le = LabelEncoder()
                y = le.fit_transform(y)
            else:
                y = y.astype(int)

            # Train/Test Split
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

            # 모델 학습
            model = LogisticRegression(max_iter=1000)
            model.fit(X_train, y_train)
            
            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test)
            if y_proba.shape[1] == 2:
                y_proba = y_proba[:, 1]
            
            # 지표 계산
            avg_mode = 'binary' if len(np.unique(y)) == 2 else 'weighted'
            
            acc = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred, average=avg_mode)
            rec = recall_score(y_test, y_pred, average=avg_mode)
            f1 = f1_score(y_test, y_pred, average=avg_mode)

            # 6-1. 지표 출력
            st.subheader("성능 지표 (Metrics)")
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Accuracy", f"{acc:.4f}")
            m2.metric("Precision", f"{prec:.4f}")
            m3.metric("Recall", f"{rec:.4f}")
            m4.metric("F1 Score", f"{f1:.4f}")

            # 6-2. 그래프 출력
            st.subheader("평가 그래프")
            g_col1, g_col2 = st.columns(2)

            with g_col1:
                st.markdown("**Confusion Matrix**")
                cm = confusion_matrix(y_test, y_pred)
                fig_cm, ax_cm = plt.subplots()
                disp = ConfusionMatrixDisplay(confusion_matrix=cm)
                disp.plot(cmap='Blues', ax=ax_cm)
                st.pyplot(fig_cm)

            with g_col2:
                st.markdown("**ROC Curve**")
                if len(np.unique(y)) == 2:
                    fpr, tpr, _ = roc_curve(y_test, y_proba)
                    roc_auc = auc(fpr, tpr)
                    fig_roc, ax_roc = plt.subplots()
                    ax_roc.plot(fpr, tpr, color='darkorange', lw=2, label=f'AUC = {roc_auc:.2f}')
                    ax_roc.plot([0, 1], [0, 1], color='navy', linestyle='--')
                    ax_roc.set_xlim([0.0, 1.0])
                    ax_roc.set_ylim([0.0, 1.05])
                    ax_roc.set_xlabel('False Positive Rate')
                    ax_roc.set_ylabel('True Positive Rate')
                    ax_roc.legend(loc="lower right")
                    st.pyplot(fig_roc)
                else:
                    st.info("다중 분류는 ROC Curve를 지원하지 않습니다.")
