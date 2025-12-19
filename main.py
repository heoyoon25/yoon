import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, confusion_matrix, roc_curve, auc, classification_report)
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# 페이지 설정
st.set_page_config(page_title="Logistic Regression App", layout="wide")

st.title("📊 로지스틱 회귀 모형 구축 (Top-Down)")

# 세션 상태 초기화 (데이터 유지를 위해 필요)
if 'df' not in st.session_state:
    st.session_state['df'] = None
if 'df_processed' not in st.session_state:
    st.session_state['df_processed'] = None
if 'selected_features' not in st.session_state:
    st.session_state['selected_features'] = []

# --- 1. 데이터 업로드 ---
st.header("1. 데이터 업로드")
uploaded_file = st.file_uploader("CSV 파일을 업로드하세요", type=["csv"])

if uploaded_file is not None:
    # 최초 업로드 시에만 데이터 로드
    if st.session_state['df'] is None:
        df = pd.read_csv(uploaded_file)
        st.session_state['df'] = df
        st.success("데이터 업로드 성공!")
    
    # 현재 데이터프레임 표시
    st.dataframe(st.session_state['df'].head())

    # --- 2. 데이터 탐색 및 시각화 ---
    st.markdown("---")
    st.header("2. 데이터 탐색 및 시각화")

    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("T-test 수행")
        # T-test를 위한 변수 선택
        t_group_col = st.selectbox("그룹 변수 (범주형/이진)", st.session_state['df'].columns, key='t_group')
        t_val_col = st.selectbox("값 변수 (수치형)", st.session_state['df'].columns, key='t_val')
        
        if st.button("T-test 실행"):
            try:
                groups = st.session_state['df'][t_group_col].unique()
                if len(groups) == 2:
                    group1 = st.session_state['df'][st.session_state['df'][t_group_col] == groups[0]][t_val_col]
                    group2 = st.session_state['df'][st.session_state['df'][t_group_col] == groups[1]][t_val_col]
                    t_stat, p_val = stats.ttest_ind(group1, group2, nan_policy='omit')
                    st.write(f"**T-statistic:** {t_stat:.4f}, **P-value:** {p_val:.4f}")
                    if p_val < 0.05:
                        st.write("결과: 통계적으로 유의미한 차이가 있습니다.")
                    else:
                        st.write("결과: 통계적으로 유의미한 차이가 없습니다.")
                else:
                    st.error("T-test는 그룹 변수의 고유값이 정확히 2개여야 합니다.")
            except Exception as e:
                st.error(f"에러 발생: {e}")

    with col2:
        st.subheader("그래프 시각화")
        viz_type = st.selectbox("그래프 유형 선택", 
                                ["Histogram", "Box Plot", "Scatter Plot", "Bar Chart", "Line Chart"])
        x_label = st.selectbox("X축 변수 선택", st.session_state['df'].columns, key='x_viz')
        y_label = st.selectbox("Y축 변수 선택 (선택사항, 히스토그램 제외)", [None] + list(st.session_state['df'].columns), key='y_viz')

        if st.button("그래프 그리기"):
            fig, ax = plt.subplots()
            try:
                if viz_type == "Histogram":
                    sns.histplot(data=st.session_state['df'], x=x_label, kde=True, ax=ax)
                elif viz_type == "Box Plot":
                    sns.boxplot(data=st.session_state['df'], x=x_label, y=y_label, ax=ax)
                elif viz_type == "Scatter Plot":
                    if y_label: sns.scatterplot(data=st.session_state['df'], x=x_label, y=y_label, ax=ax)
                    else: st.warning("Scatter Plot은 Y축 변수가 필요합니다.")
                elif viz_type == "Bar Chart":
                    if y_label: sns.barplot(data=st.session_state['df'], x=x_label, y=y_label, ax=ax)
                    else: st.session_state['df'][x_label].value_counts().plot(kind='bar', ax=ax)
                elif viz_type == "Line Chart":
                    if y_label: sns.lineplot(data=st.session_state['df'], x=x_label, y=y_label, ax=ax)
                    else: st.warning("Line Chart는 Y축 변수가 필요합니다.")
                
                st.pyplot(fig)
            except Exception as e:
                st.error(f"그래프 오류: {e}")

    # --- 3. 데이터 전처리 ---
    st.markdown("---")
    st.header("3. 데이터 전처리")
    
    # 3-1. 변수 선택 (Target 설정을 먼저 해야 전처리가 명확해짐)
    st.subheader("변수 설정")
    all_columns = st.session_state['df'].columns.tolist()
    target_variable = st.selectbox("종속 변수 (Target, Y) 선택", all_columns)
    feature_variables = st.multiselect("독립 변수 (Features, X) 선택", [c for c in all_columns if c != target_variable])

    # 3-2. 전처리 실행 버튼
    if st.button("데이터 전처리 실행 (결측치, 스케일링, 인코딩)"):
        if not feature_variables or not target_variable:
            st.error("독립변수와 종속변수를 먼저 선택해주세요.")
        else:
            df_curr = st.session_state['df'].copy()
            X = df_curr[feature_variables]
            y = df_curr[target_variable]

            # 수치형/범주형 구분
            numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
            categorical_features = X.select_dtypes(include=['object', 'category']).columns

            # 전처리 파이프라인
            numeric_transformer = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='mean')), # 결측치 평균 대치
                ('scaler', StandardScaler()) # 스케일링
            ])

            categorical_transformer = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='most_frequent')), # 최빈값 대치
                ('onehot', OneHotEncoder(handle_unknown='ignore')) # 원핫인코딩
            ])

            preprocessor = ColumnTransformer(
                transformers=[
                    ('num', numeric_transformer, numeric_features),
                    ('cat', categorical_transformer, categorical_features)
                ])

            # 전처리 수행
            try:
                X_processed = preprocessor.fit_transform(X)
                
                # 컬럼 이름 복원 (OneHotEncoder 등 반영)
                new_cols = []
                if len(numeric_features) > 0:
                    new_cols.extend(numeric_features)
                if len(categorical_features) > 0:
                     # OneHotEncoder의 feature name 가져오기
                    cat_encoder = preprocessor.named_transformers_['cat']['onehot']
                    new_cols.extend(cat_encoder.get_feature_names_out(categorical_features))
                
                # DataFrame으로 변환
                X_processed_df = pd.DataFrame(X_processed, columns=new_cols)
                
                # 이상치 처리 (간단하게 IQR 방식으로 필터링은 생략하고 스케일링으로 대체하거나, 필요시 추가 구현)
                # 여기서는 Target 변수의 결측치 제거만 수행
                y = y.fillna(y.mode()[0])
                
                # 세션에 저장
                st.session_state['df_processed'] = pd.concat([X_processed_df, y.reset_index(drop=True)], axis=1)
                st.session_state['X_final'] = X_processed_df
                st.session_state['y_final'] = y.reset_index(drop=True)
                
                st.success("전처리가 완료되었습니다.")
                st.dataframe(st.session_state['df_processed'].head())
                
            except Exception as e:
                st.error(f"전처리 중 오류 발생: {e}")

    # 3-3. Stepwise Selection (전진 선택법)
    if st.button("Stepwise Selection (전진 선택법)"):
        if 'X_final' in st.session_state and st.session_state['X_final'] is not None:
            with st.spinner("변수 선택 중입니다..."):
                try:
                    selector = SequentialFeatureSelector(
                        LogisticRegression(max_iter=1000),
                        direction='forward',
                        n_features_to_select='auto',
                        tol=None,
                        scoring='accuracy',
                        cv=3
                    )
                    selector.fit(st.session_state['X_final'], st.session_state['y_final'])
                    selected_mask = selector.get_support()
                    selected_feats = st.session_state['X_final'].columns[selected_mask].tolist()
                    
                    st.session_state['selected_features'] = selected_feats
                    st.success(f"선택된 변수: {selected_feats}")
                except Exception as e:
                    st.error(f"변수 선택 오류: {e}")
        else:
            st.warning("먼저 '데이터 전처리'를 실행해주세요.")

    # 3-4. 데이터 나누기
    st.subheader("데이터 나누기 (Train/Test Split)")
    test_size = st.slider("테스트 데이터 비율 설정", 0.1, 0.5, 0.2)

    # --- 4. 모형 구축 및 평가 ---
    st.markdown("---")
    st.header("4. 모형 구축 및 평가")

    if st.button("로지스틱 회귀 모델 학습 및 평가"):
        if 'X_final' not in st.session_state or st.session_state['X_final'] is None:
             st.error("데이터 전처리가 선행되어야 합니다.")
        else:
            # Stepwise로 선택된 변수가 있으면 그것만 사용, 없으면 전체 사용
            features_to_use = st.session_state['selected_features'] if st.session_state['selected_features'] else st.session_state['X_final'].columns.tolist()
            
            X_model = st.session_state['X_final'][features_to_use]
            y_model = st.session_state['y_final']

            # Train/Test Split
            X_train, X_test, y_train, y_test = train_test_split(X_model, y_model, test_size=test_size, random_state=42)

            # 모델 학습
            model = LogisticRegression(max_iter=1000)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test)[:, 1] # ROC용 확률

            # 4-1. 평가지표 출력
            st.subheader("모델 성능 지표")
            col_m1, col_m2, col_m3, col_m4 = st.columns(4)
            col_m1.metric("Accuracy", f"{accuracy_score(y_test, y_pred):.4f}")
            col_m2.metric("Precision", f"{precision_score(y_test, y_pred, average='weighted', zero_division=0):.4f}")
            col_m3.metric("Recall", f"{recall_score(y_test, y_pred, average='weighted', zero_division=0):.4f}")
            col_m4.metric("F1-Score", f"{f1_score(y_test, y_pred, average='weighted', zero_division=0):.4f}")

            # 4-2. 시각화 (Confusion Matrix & ROC Curve)
            col_v1, col_v2 = st.columns(2)

            with col_v1:
                st.subheader("Confusion Matrix")
                cm = confusion_matrix(y_test, y_pred)
                fig_cm, ax_cm = plt.subplots()
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax_cm)
                ax_cm.set_xlabel('Predicted')
                ax_cm.set_ylabel('Actual')
                st.pyplot(fig_cm)

            with col_v2:
                st.subheader("ROC Curve")
                # 이진 분류일 때만 ROC Curve가 의미가 있습니다. 다중 분류인 경우 처리가 필요합니다.
                if len(np.unique(y_model)) == 2:
                    fpr, tpr, thresholds = roc_curve(y_test, y_proba)
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
                else:
                    st.info("ROC Curve는 이진 분류(타겟 클래스가 2개)일 때만 표시됩니다.")

else:
    st.info("CSV 파일을 업로드하면 분석이 시작됩니다.")
