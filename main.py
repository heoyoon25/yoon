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

# 세션 상태 초기화 (데이터 유지용)
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

# 데이터가 있을 경우에만 이후 단계 진행
if st.session_state['df'] is not None:
    df = st.session_state['df']

    st.markdown("---")
    # ==========================================
    # 2. 데이터 탐색 및 시각화
    # ==========================================
    st.header("2. 데이터 탐색 및 시각화 (EDA)")

    # 2-1. T-test (p-value <= 0.05)
    st.subheader("가설 검정 (T-test)")
    st.caption("이진 그룹(0/1 등)에 따른 수치형 변수의 평균 차이를 검정합니다.")
    
    col1, col2 = st.columns(2)
    with col1:
        group_col = st.selectbox("그룹 변수 (이진 범주형)", df.columns, key='ttest_group')
    with col2:
        target_num_col = st.selectbox("검정할 수치형 변수", df.select_dtypes(include=np.number).columns, key='ttest_val')

    if st.button("T-test 실행"):
        try:
            groups = df[group_col].unique()
            if len(groups) != 2:
                st.error("그룹 변수는 정확히 2개의 고유값(예: 0과 1)을 가져야 합니다.")
            else:
                group_a = df[df[group_col] == groups[0]][target_num_col]
                group_b = df[df[group_col] == groups[1]][target_num_col]
                
                t_stat, p_val = stats.ttest_ind(group_a, group_b, equal_var=False) # Welch's t-test
                
                st.write(f"**T-statistic:** {t_stat:.4f}, **P-value:** {p_val:.4f}")
                
                if p_val <= 0.05:
                    st.success(f"P-value가 {p_val:.4f}로 0.05 이하입니다. 유의미한 차이가 있습니다.")
                else:
                    st.warning(f"P-value가 {p_val:.4f}로 0.05보다 큽니다. 유의미한 차이가 없습니다.")
        except Exception as e:
            st.error(f"오류 발생: {e}")

    # 2-2. 시각화
    st.subheader("그래프 시각화")
    v_col1, v_col2, v_col3 = st.columns(3)
    
    with v_col1:
        x_axis = st.selectbox("X축 선택", df.columns)
    with v_col2:
        y_axis = st.selectbox("Y축 선택 (필요 시)", [None] + list(df.columns))
    with v_col3:
        plot_type = st.selectbox("그래프 유형", 
                                 ["Histogram", "Box Plot", "Scatter Plot", "Bar Chart", "Line Chart"])

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
                st.warning("Scatter Plot은 Y축 선택이 필요합니다.")
        elif plot_type == "Bar Chart":
            if y_axis:
                sns.barplot(data=df, x=x_axis, y=y_axis, ax=ax)
            else:
                st.countplot(data=df, x=x_axis, ax=ax)
        elif plot_type == "Line Chart":
            if y_axis:
                sns.lineplot(data=df, x=x_axis, y=y_axis, ax=ax)
            else:
                st.warning("Line Chart는 Y축 선택이 필요합니다.")
        
        st.pyplot(fig)
    except Exception as e:
        st.error(f"그래프를 그리는 중 오류가 발생했습니다: {e}")

    st.markdown("---")
    # ==========================================
    # 3. 데이터 전처리
    # ==========================================
    st.header("3. 데이터 전처리")
    
    # 전처리 옵션 선택
    col_p1, col_p2, col_p3 = st.columns(3)
    with col_p1:
        handle_na = st.checkbox("결측치 제거 (Drop NA)")
    with col_p2:
        do_scaling = st.checkbox("특성 스케일링 (StandardScaler)")
    with col_p3:
        do_encoding = st.checkbox("원-핫 인코딩 (범주형 변수 변환)")

    if st.button("데이터 전처리 적용"):
        df_processed = df.copy()
        
        # 1. 결측치 처리
        if handle_na:
            df_processed = df_processed.dropna()
            st.info("결측치를 제거했습니다.")

        # 2. 원-핫 인코딩 (수치형이 아닌 컬럼 대상)
        if do_encoding:
            cat_cols = df_processed.select_dtypes(include=['object', 'category']).columns
            if len(cat_cols) > 0:
                df_processed = pd.get_dummies(df_processed, columns=cat_cols, drop_first=True)
                st.info(f"원-핫 인코딩 완료: {list(cat_cols)}")
        
        # 3. 스케일링 (수치형 컬럼 대상, 타겟 변수는 제외해야 하므로 주의 필요 - 여기서는 전체 적용 후 모델링 단계에서 분리 권장하지만, 단순화를 위해 수치형만 변환)
        if do_scaling:
            num_cols = df_processed.select_dtypes(include=np.number).columns
            scaler = StandardScaler()
            df_processed[num_cols] = scaler.fit_transform(df_processed[num_cols])
            st.info("스케일링(StandardScaler)을 적용했습니다.")

        # 전처리된 데이터를 세션에 업데이트
        st.session_state['df_processed'] = df_processed
        st.success("전처리 완료!")
        st.write(st.session_state['df_processed'].head())

    # 전처리된 데이터가 있으면 그것을 사용, 없으면 원본 사용
    current_df = st.session_state.get('df_processed', df)

    st.markdown("---")
    # ==========================================
    # 5. 데이터 나누기 & 변수 선택 (순서 조정: 선택 후 Stepwise 적용)
    # ==========================================
    st.header("4 & 5. 변수 선택 및 데이터 나누기")
    
    # 타겟 변수 선택
    target_col = st.selectbox("종속 변수 (Target) 선택", current_df.columns)
    
    # 독립 변수 후보군 선택
    feature_candidates = [c for c in current_df.columns if c != target_col]
    selected_features = st.multiselect("독립 변수 (Features) 선택", feature_candidates, default=feature_candidates)

    # 4. 특성 선택 (Stepwise Selection - Forward)
    st.subheader("4. 특성 선택 (Stepwise Selection - Forward)")
    
    if st.button("전진 선택법(Forward Stepwise) 실행"):
        if not selected_features:
            st.warning("먼저 독립 변수를 선택해주세요.")
        else:
            X_temp = current_df[selected_features]
            y_temp = current_df[target_col]
            
            # y가 연속형이면 안되므로 라벨 인코딩 체크 (로지스틱 회귀용)
            if y_temp.dtype == 'object':
                le = LabelEncoder()
                y_temp = le.fit_transform(y_temp)

            try:
                model = LogisticRegression(solver='liblinear')
                # n_features_to_select='auto'로 두면 절반 정도를 선택함. 여기선 50% 선택으로 설정
                sfs = SequentialFeatureSelector(model, direction='forward', n_features_to_select='auto', tol=None)
                
                with st.spinner("최적의 변수를 찾는 중입니다..."):
                    sfs.fit(X_temp, y_temp)
                
                selected_mask = sfs.get_support()
                suggested_features = np.array(selected_features)[selected_mask]
                
                st.success(f"선택된 변수 ({len(suggested_features)}개): {', '.join(suggested_features)}")
                # 선택된 변수로 업데이트할지 여부는 사용자 판단에 맡기거나 자동으로 multiselect에 반영할 수 있음
                st.info("위 변수들을 참고하여 아래 '최종 변수 선택'을 조정하세요.")
            
            except Exception as e:
                st.error(f"Stepwise 실행 중 오류: {e}")

    # 데이터 나누기 설정
    st.subheader("5. 데이터 분할 설정")
    test_size = st.slider("테스트 데이터 비율 (Test Size)", 0.1, 0.5, 0.2)

    st.markdown("---")
    # ==========================================
    # 6. 모형 구축 및 평가
    # ==========================================
    st.header("6. 모형 구축 및 평가")

    if st.button("로지스틱 회귀 모델 학습 시작"):
        if not selected_features:
            st.error("독립 변수를 하나 이상 선택해야 합니다.")
        else:
            # 데이터 준비
            X = current_df[selected_features]
            y = current_df[target_col]

            # 타겟 변수 인코딩 (필요 시)
            if y.dtype == 'object' or y.dtype.name == 'category':
                le = LabelEncoder()
                y = le.fit_transform(y)
            else:
                # 0과 1인지 확인, 아니면 변환 시도
                if len(np.unique(y)) > 2:
                     st.warning("경고: 종속 변수의 클래스가 2개 이상입니다. 다중 분류(Multinomial)로 처리됩니다.")
                y = y.astype(int)

            # Train/Test Split
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

            # 모델 학습
            model = LogisticRegression(max_iter=1000)
            model.fit(X_train, y_train)
            
            # 예측
            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test)
            
            # 이진 분류일 경우 확률의 두 번째 컬럼(Class 1) 사용
            if y_proba.shape[1] == 2:
                y_proba = y_proba[:, 1]
            
            # 6-1. 평가지표 출력
            st.subheader("모델 성능 지표")
            col_m1, col_m2, col_m3, col_m4 = st.columns(4)
            
            # Average 매개변수는 이진 분류 기본값인 'binary' 사용하되, 다중 분류일 경우 'weighted' 적용
            avg_method = 'binary' if len(np.unique(y)) == 2 else 'weighted'
            
            col_m1.metric("Accuracy", f"{accuracy_score(y_test, y_pred):.4f}")
            col_m2.metric("Precision", f"{precision_score(y_test, y_pred, average=avg_method):.4f}")
            col_m3.metric("Recall", f"{recall_score(y_test, y_pred, average=avg_method):.4f}")
            col_m4.metric("F1 Score", f"{f1_score(y_test, y_pred, average=avg_method):.4f}")

            # 6-2. 시각화 (ROC, Confusion Matrix)
            st.subheader("모델 평가 시각화")
            plot_col1, plot_col2 = st.columns(2)

            # Confusion Matrix
            with plot_col1:
                st.write("**Confusion Matrix**")
                cm = confusion_matrix(y_test, y_pred)
                fig_cm, ax_cm = plt.subplots()
                disp = ConfusionMatrixDisplay(confusion_matrix=cm)
                disp.plot(cmap='Blues', ax=ax_cm)
                st.pyplot(fig_cm)

            # ROC Curve (이진 분류인 경우에만)
            with plot_col2:
                st.write("**ROC Curve**")
                if len(np.unique(y)) == 2:
                    fpr, tpr, thresholds = roc_curve(y_test, y_proba)
                    roc_auc = auc(fpr, tpr)
                    
                    fig_roc, ax_roc = plt.subplots()
                    ax_roc.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
                    ax_roc.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
                    ax_roc.set_xlim([0.0, 1.0])
                    ax_roc.set_ylim([0.0, 1.05])
                    ax_roc.set_xlabel('False Positive Rate')
                    ax_roc.set_ylabel('True Positive Rate')
                    ax_roc.legend(loc="lower right")
                    st.pyplot(fig_roc)
                else:
                    st.info("다중 분류 문제에서는 ROC Curve가 단순 2차원 플롯으로 제공되지 않습니다.")
