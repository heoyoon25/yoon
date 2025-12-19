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

# 페이지 설정
st.set_page_config(page_title="로짓 모형 분석기 (Fixed Y)", layout="wide")

st.title("📊 Logistic Regression Tool (Target: not.fully.paid)")
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
    
    # 데이터 미리보기
    st.dataframe(st.session_state['df'].head())
    
    # not.fully.paid 컬럼 존재 여부 확인
    if 'not.fully.paid' not in st.session_state['df'].columns:
        st.error("⚠️ 경고: 업로드된 데이터에 'not.fully.paid' 변수가 없습니다. 컬럼명을 확인해주세요.")

# 데이터가 있을 때만 실행
if st.session_state['df'] is not None and 'not.fully.paid' in st.session_state['df'].columns:
    df = st.session_state['df']
    
    # [중요] 종속 변수 고정
    target_col = 'not.fully.paid'

    st.markdown("---")
    # ==========================================
    # 2. 데이터 탐색 및 시각화
    # ==========================================
    st.header("2. 데이터 탐색 및 시각화 (EDA)")

    # 2-1. T-test (자동)
    st.subheader(f"가설 검정 (Target: {target_col} 기준)")
    st.caption(f"'{target_col}'(0/1)에 따라 평균 차이가 유의미한(p<=0.05) 변수만 추출합니다.")

    if st.button("유의한 변수 찾기 (T-test)"):
        # 그룹 변수는 고정된 target_col 사용
        if df[target_col].nunique() != 2:
            st.error(f"오류: '{target_col}' 변수의 값이 2개(0과 1)가 아닙니다.")
        else:
            # 수치형 변수만 선택 (Target 제외)
            numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
            if target_col in numeric_cols:
                numeric_cols.remove(target_col)
            
            significant_vars = []
            groups = df[target_col].unique()
            
            for col in numeric_cols:
                try:
                    group_a = df[df[target_col] == groups[0]][col]
                    group_b = df[df[target_col] == groups[1]][col]
                    
                    t_stat, p_val = stats.ttest_ind(group_a, group_b, equal_var=False, nan_policy='omit')
                    
                    if p_val <= 0.05:
                        significant_vars.append({
                            "변수명": col,
                            "T-statistic": round(t_stat, 4),
                            "P-value": round(p_val, 5)
                        })
                except:
                    pass

            if significant_vars:
                st.success(f"유의미한 변수 {len(significant_vars)}개 발견")
                st.dataframe(pd.DataFrame(significant_vars))
            else:
                st.warning("P-value <= 0.05인 변수가 없습니다.")

    st.markdown("---")

    # 2-2. 시각화
    st.subheader("그래프 시각화")
    v_col1, v_col2, v_col3 = st.columns(3)
    
    with v_col1:
        x_axis = st.selectbox("X축 선택", df.columns)
    with v_col2:
        y_axis = st.selectbox("Y축 선택 (선택)", [None] + list(df.columns))
    with v_col3:
        plot_type = st.selectbox("그래프 유형", 
                                 ["Histogram", "Box Plot", "Scatter Plot", "Bar Chart", "Line Chart"])

    if st.button("그래프 그리기"):
        fig, ax = plt.subplots(figsize=(10, 5))
        try:
            if plot_type == "Histogram":
                sns.histplot(data=df, x=x_axis, hue=target_col, kde=True, ax=ax) # hue에 타겟 적용하여 구분
            elif plot_type == "Box Plot":
                sns.boxplot(data=df, x=x_axis, y=y_axis, ax=ax)
            elif plot_type == "Scatter Plot":
                if y_axis: sns.scatterplot(data=df, x=x_axis, y=y_axis, hue=target_col, ax=ax)
                else: st.warning("Y축을 선택하세요.")
            elif plot_type == "Bar Chart":
                if y_axis: sns.barplot(data=df, x=x_axis, y=y_axis, ax=ax)
                else: sns.countplot(data=df, x=x_axis, ax=ax)
            elif plot_type == "Line Chart":
                if y_axis: sns.lineplot(data=df, x=x_axis, y=y_axis, ax=ax)
                else: st.warning("Y축을 선택하세요.")
            
            st.pyplot(fig)
        except Exception as e:
            st.error(f"오류: {e}")

    st.markdown("---")
    # ==========================================
    # 3. 데이터 전처리
    # ==========================================
    st.header("3. 데이터 전처리")
    
    c1, c2, c3 = st.columns(3)
    handle_na = c1.checkbox("결측치 제거")
    do_scaling = c2.checkbox("스케일링 (StandardScaler)")
    do_encoding = c3.checkbox("원-핫 인코딩")

    if st.button("전처리 적용"):
        df_proc = df.copy()
        
        if handle_na:
            df_proc = df_proc.dropna()
        
        if do_encoding:
            cat_cols = df_proc.select_dtypes(include=['object', 'category']).columns
            if len(cat_cols) > 0:
                df_proc = pd.get_dummies(df_proc, columns=cat_cols, drop_first=True)
        
        if do_scaling:
            # 타겟 변수 제외하고 스케일링
            num_cols = df_proc.select_dtypes(include=np.number).columns.tolist()
            if target_col in num_cols:
                num_cols.remove(target_col)
            
            scaler = StandardScaler()
            df_proc[num_cols] = scaler.fit_transform(df_proc[num_cols])

        st.session_state['df_processed'] = df_proc
        st.success("전처리 완료")
        st.dataframe(df_proc.head())

    current_df = st.session_state.get('df_processed', df)

    st.markdown("---")
    # ==========================================
    # 4. 특성 선택 (Stepwise) - 오류 수정됨
    # ==========================================
    st.header("4. 특성 선택 (Stepwise Selection)")
    
    st.info(f"📍 종속 변수(Y)는 **'{target_col}'**로 고정되어 있습니다.")

    # 독립 변수 후보군 (타겟 제외)
    feature_candidates = [c for c in current_df.columns if c != target_col]
    selected_features_pool = st.multiselect("Stepwise 후보 변수 선택", feature_candidates, default=feature_candidates)

    if st.button("전진 선택법(Stepwise) 실행"):
        if not selected_features_pool:
            st.warning("변수를 선택하세요.")
        else:
            # 1. X, y 준비
            X_temp = current_df[selected_features_pool]
            y_temp = current_df[target_col]

            # [핵심 수정] y를 무조건 정수형(Label)으로 변환하여 'continuous' 오류 방지
            le = LabelEncoder()
            y_encoded = le.fit_transform(y_temp)

            try:
                # 2. 모델 설정
                model_sel = LogisticRegression(solver='liblinear') 
                
                # 3. Stepwise 실행
                sfs = SequentialFeatureSelector(model_sel, direction='forward', n_features_to_select='auto')
                
                with st.spinner("최적 변수 탐색 중..."):
                    sfs.fit(X_temp, y_encoded) # 인코딩된 y 사용
                
                # 4. 결과 도출
                selected_mask = sfs.get_support()
                recommended_features = np.array(selected_features_pool)[selected_mask]
                
                st.success(f"추천 변수 ({len(recommended_features)}개): {', '.join(recommended_features)}")
                
            except Exception as e:
                st.error(f"오류 발생: {e}")

    st.markdown("---")
    # ==========================================
    # 5 & 6. 데이터 나누기 / 모델 구축
    # ==========================================
    st.header("5 & 6. 최종 변수 선택 및 모델 평가")

    c_final1, c_final2 = st.columns(2)
    final_features = c_final1.multiselect("최종 독립 변수 선택", feature_candidates)
    test_size = c_final2.slider("Test Size", 0.1, 0.5, 0.2)

    if st.button("모델 학습 및 평가"):
        if not final_features:
            st.error("변수를 선택하세요.")
        else:
            X = current_df[final_features]
            y = current_df[target_col]

            # [핵심 수정] 학습시에도 안전하게 인코딩 적용
            le_final = LabelEncoder()
            y_encoded_final = le_final.fit_transform(y)

            # Split
            X_train, X_test, y_train, y_test = train_test_split(X, y_encoded_final, test_size=test_size, random_state=42)

            # Model Fit
            model = LogisticRegression(max_iter=3000)
            model.fit(X_train, y_train)
            
            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test)[:, 1]

            # Metrics
            st.subheader("모델 성능")
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Accuracy", f"{accuracy_score(y_test, y_pred):.4f}")
            m2.metric("Precision", f"{precision_score(y_test, y_pred, zero_division=0):.4f}")
            m3.metric("Recall", f"{recall_score(y_test, y_pred, zero_division=0):.4f}")
            m4.metric("F1 Score", f"{f1_score(y_test, y_pred, zero_division=0):.4f}")

            # Plots
            st.subheader("시각화")
            gc1, gc2 = st.columns(2)
            
            with gc1:
                st.write("**Confusion Matrix**")
                cm = confusion_matrix(y_test, y_pred)
                disp = ConfusionMatrixDisplay(cm)
                fig_cm, ax_cm = plt.subplots()
                disp.plot(cmap='Blues', ax=ax_cm)
                st.pyplot(fig_cm)
            
            with gc2:
                st.write("**ROC Curve**")
                fpr, tpr, _ = roc_curve(y_test, y_proba)
                roc_auc = auc(fpr, tpr)
                fig_roc, ax_roc = plt.subplots()
                ax_roc.plot(fpr, tpr, color='orange', label=f'AUC = {roc_auc:.2f}')
                ax_roc.plot([0,1],[0,1], 'k--')
                ax_roc.legend()
                st.pyplot(fig_roc)
