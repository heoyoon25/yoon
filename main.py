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
st.set_page_config(page_title="로짓 모형 분석", layout="wide")

st.title("📊 Logistic Regression Tool (T-test -> Stepwise Link)")
st.markdown("---")

# ==========================================
# 세션 상태 초기화
# ==========================================
if 'df' not in st.session_state:
    st.session_state['df'] = None
if 'recommended_features' not in st.session_state:
    st.session_state['recommended_features'] = [] # Stepwise 결과 저장용
if 'significant_features' not in st.session_state:
    st.session_state['significant_features'] = [] # T-test 결과 저장용 (NEW)

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
    
    if 'not.fully.paid' not in st.session_state['df'].columns:
        st.error("⚠️ 경고: 업로드된 데이터에 'not.fully.paid' 변수가 없습니다.")

# 데이터가 있을 때만 실행
if st.session_state['df'] is not None and 'not.fully.paid' in st.session_state['df'].columns:
    df = st.session_state['df']
    target_col = 'not.fully.paid'

    st.markdown("---")
    # ==========================================
    # 2. 데이터 탐색 (T-test 로직 수정됨)
    # ==========================================
    st.header("2. 데이터 탐색 및 시각화 (EDA)")

    st.subheader(f"가설 검정 (Target: {target_col} 기준)")
    st.caption(f"'{target_col}'(0/1)에 따라 평균 차이가 유의미한(p<=0.05) 변수만 추출하여 **Stepwise 후보로 등록합니다.**")

    if st.button("유의한 변수 찾기 (T-test)"):
        if df[target_col].nunique() != 2:
            st.error(f"오류: '{target_col}' 변수의 값이 2개(0과 1)가 아닙니다.")
        else:
            numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
            if target_col in numeric_cols:
                numeric_cols.remove(target_col)
            
            significant_vars = []
            sig_names_temp = [] # 변수명만 저장할 리스트
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
                        sig_names_temp.append(col)
                except:
                    pass

            if significant_vars:
                # [핵심] 발견된 유의미한 변수들을 세션에 저장
                st.session_state['significant_features'] = sig_names_temp
                
                st.success(f"유의미한 변수 {len(significant_vars)}개 발견 및 저장 완료!")
                st.dataframe(pd.DataFrame(significant_vars))
                st.info("👇 이 변수들이 4번 Stepwise 선택 단계의 '후보 변수'로 자동 설정됩니다.")
            else:
                st.warning("P-value <= 0.05인 변수가 없습니다.")
                st.session_state['significant_features'] = []

    st.markdown("---")
    # 2-2. 시각화 (생략 없이 유지)
    st.subheader("그래프 시각화")
    v_col1, v_col2, v_col3 = st.columns(3)
    with v_col1: x_axis = st.selectbox("X축 선택", df.columns)
    with v_col2: y_axis = st.selectbox("Y축 선택 (선택)", [None] + list(df.columns))
    with v_col3: plot_type = st.selectbox("그래프 유형", ["Histogram", "Box Plot", "Scatter Plot", "Bar Chart", "Line Chart"])

    if st.button("그래프 그리기"):
        fig, ax = plt.subplots(figsize=(10, 5))
        try:
            if plot_type == "Histogram": sns.histplot(data=df, x=x_axis, hue=target_col, kde=True, ax=ax)
            elif plot_type == "Box Plot": sns.boxplot(data=df, x=x_axis, y=y_axis, ax=ax)
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
        except Exception as e: st.error(f"오류: {e}")

    st.markdown("---")
    # ==========================================
    # 3. 데이터 전처리
    # ==========================================
    st.header("3. 데이터 전처리")
    
    c1, c2, c3 = st.columns(3)
    handle_na = c1.checkbox("결측치 제거", value=True)
    do_scaling = c2.checkbox("스케일링 (StandardScaler)")
    do_encoding = c3.checkbox("원-핫 인코딩")

    if st.button("전처리 적용"):
        df_proc = df.copy()
        if handle_na: df_proc = df_proc.dropna()
        
        if do_encoding:
            cat_cols = df_proc.select_dtypes(include=['object', 'category']).columns
            if len(cat_cols) > 0:
                df_proc = pd.get_dummies(df_proc, columns=cat_cols, drop_first=True)
        
        if do_scaling:
            num_cols = df_proc.select_dtypes(include=np.number).columns.tolist()
            if target_col in num_cols: num_cols.remove(target_col)
            scaler = StandardScaler()
            df_proc[num_cols] = scaler.fit_transform(df_proc[num_cols])

        st.session_state['df_processed'] = df_proc
        st.session_state['recommended_features'] = [] 
        st.success("전처리 완료")
        st.dataframe(df_proc.head())

    current_df = st.session_state.get('df_processed', df)

    st.markdown("---")
    # ==========================================
    # 4. 특성 선택 (Stepwise) - T-test 연동 수정됨
    # ==========================================
    st.header("4. 특성 선택 (Stepwise Selection)")
    st.info(f"📍 종속 변수(Y): **'{target_col}'**")

    # 독립 변수 후보군 (전체 컬럼 중 타겟 제외)
    feature_candidates = [c for c in current_df.columns if c != target_col]

    # [핵심] Default 값 결정 로직
    # 1순위: T-test에서 유의하다고 판명된 변수들 (st.session_state['significant_features'])
    # 2순위: T-test를 안 돌렸다면 전체 변수
    
    default_candidates = []
    
    if st.session_state['significant_features']:
        # T-test 변수 중 현재 데이터프레임(전처리 후)에 실제로 존재하는 것만 필터링
        default_candidates = [f for f in st.session_state['significant_features'] if f in feature_candidates]
        st.success(f"✅ T-test 검정 결과, 유의미한 변수 {len(default_candidates)}개가 기본 선택되었습니다.")
    else:
        # T-test 안 돌렸으면 전체 선택
        default_candidates = feature_candidates

    selected_features_pool = st.multiselect(
        "Stepwise 후보 변수 선택", 
        options=feature_candidates, 
        default=default_candidates
    )

    if st.button("전진 선택법(Stepwise) 실행"):
        if not selected_features_pool:
            st.warning("변수를 선택하세요.")
        else:
            X_temp = current_df[selected_features_pool]
            y_temp = current_df[target_col]
            
            le = LabelEncoder()
            y_encoded = le.fit_transform(y_temp)

            try:
                model_sel = LogisticRegression(solver='liblinear') 
                sfs = SequentialFeatureSelector(model_sel, direction='forward', n_features_to_select='auto')
                
                with st.spinner("최적 변수 탐색 중..."):
                    sfs.fit(X_temp, y_encoded)
                
                selected_mask = sfs.get_support()
                recommended = np.array(selected_features_pool)[selected_mask]
                
                # 결과 저장
                st.session_state['recommended_features'] = list(recommended)
                st.success(f"추천 변수 ({len(recommended)}개): {', '.join(recommended)}")
                st.info("👇 아래 '최종 독립 변수 선택' 란에 자동으로 반영되었습니다.")
                
            except Exception as e:
                st.error(f"오류 발생: {e}")


    # ... (4번 Stepwise 까지 동일) ...

    st.markdown("---")
    # ==========================================
    # 5 & 6. 최종 모델링 (샘플링 방식 선택 기능 추가)
    # ==========================================
    st.header("5 & 6. 최종 변수 선택 및 모델 평가")

    # 라이브러리 로드
    try:
        from imblearn.over_sampling import SMOTE
        from imblearn.under_sampling import RandomUnderSampler
    except Exception as e:
        st.error(f"라이브러리 로드 실패: {e}")
        st.stop()

    c_final1, c_final2 = st.columns(2)

    # Stepwise 결과가 있으면 Default로 사용
    final_default = [f for f in st.session_state.get('recommended_features', []) if f in feature_candidates]
    
    final_features = c_final1.multiselect(
        "최종 독립 변수 선택", 
        options=feature_candidates,
        default=final_default
    )
    
    test_size = c_final2.slider("Test Size", 0.1, 0.5, 0.2)

    st.subheader("⚙️ 불균형 데이터 처리 옵션")
    h1, h2, h3 = st.columns(3)
    
    # [NEW] 샘플링 방식 선택 라디오 버튼
    sampling_method = h1.radio("샘플링 방식 선택", ["적용 안 함", "SMOTE (오버샘플링)", "Under-sampling (언더샘플링)"])
    
    threshold = h2.slider("분류 임계값 (Threshold)", 0.0, 1.0, 0.5, 0.01)

    if st.button("모델 학습 및 평가"):
        if not final_features:
            st.error("변수를 선택하세요.")
        else:
            X = current_df[final_features]
            y = current_df[target_col]

            le_final = LabelEncoder()
            y_encoded_final = le_final.fit_transform(y)

            # 1. Train/Test Split
            X_train, X_test, y_train, y_test = train_test_split(X, y_encoded_final, test_size=test_size, random_state=42)

            # 2. 샘플링 적용 (학습 데이터에만!)
            if sampling_method == "SMOTE (오버샘플링)":
                sampler = SMOTE(random_state=42)
                X_train_res, y_train_res = sampler.fit_resample(X_train, y_train)
                st.info(f"⚡ SMOTE 적용: 데이터가 {len(y_train)}개 -> {len(y_train_res)}개로 늘어났습니다.")
                
            elif sampling_method == "Under-sampling (언더샘플링)":
                sampler = RandomUnderSampler(random_state=42)
                X_train_res, y_train_res = sampler.fit_resample(X_train, y_train)
                st.warning(f"✂️ 언더샘플링 적용: 데이터가 {len(y_train)}개 -> {len(y_train_res)}개로 줄어들었습니다.")
                
            else: # 적용 안 함
                X_train_res, y_train_res = X_train, y_train
                st.text("샘플링 미적용 (원본 데이터 사용)")

            # 3. 모델 학습
            model = LogisticRegression(max_iter=5000)
            model.fit(X_train_res, y_train_res)
            
            # 4. 예측
            y_proba = model.predict_proba(X_test)[:, 1]
            y_pred = (y_proba >= threshold).astype(int)

            # --- 결과 출력 ---
            st.subheader("모델 성능")
            
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Accuracy", f"{accuracy_score(y_test, y_pred):.4f}")
            m2.metric("Precision", f"{precision_score(y_test, y_pred, zero_division=0):.4f}")
            m3.metric("Recall", f"{recall_score(y_test, y_pred, zero_division=0):.4f}")
            m4.metric("F1 Score", f"{f1_score(y_test, y_pred, zero_division=0):.4f}")

            # --- 시각화 ---
            st.subheader("시각화 및 진단")
            
            # 히스토그램
            st.write("#### 1. 예측 확률 분포")
            fig_hist, ax_hist = plt.subplots(figsize=(10, 3))
            sns.histplot(y_proba, bins=50, kde=True, ax=ax_hist, color='skyblue')
            ax_hist.axvline(threshold, color='red', linestyle='--', label=f'Threshold: {threshold}')
            ax_hist.legend()
            st.pyplot(fig_hist)
            
            # Confusion Matrix
            gc1, gc2 = st.columns(2)
            with gc1:
                st.write("#### 2. Confusion Matrix")
                cm = confusion_matrix(y_test, y_pred)
                fig_cm, ax_cm = plt.subplots()
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax_cm, annot_kws={"size": 14})
                ax_cm.set_xlabel('Predicted Label')
                ax_cm.set_ylabel('True Label')
                st.pyplot(fig_cm)
            
            with gc2:
                st.write("#### 3. ROC Curve")
                fpr, tpr, _ = roc_curve(y_test, y_proba)
                roc_auc = auc(fpr, tpr)
                fig_roc, ax_roc = plt.subplots()
                ax_roc.plot(fpr, tpr, color='orange', label=f'AUC = {roc_auc:.2f}')
                ax_roc.plot([0,1],[0,1], 'k--')
                ax_roc.legend()
                st.pyplot(fig_roc)

# ... (ROC Curve 그리는 코드 아래에 추가) ...

            # [중요] 7번 파트에서 쓰기 위해 결과 저장
            st.session_state['final_model'] = model
            st.session_state['y_test_final'] = y_test
            st.session_state['y_proba_final'] = y_proba
            st.session_state['y_pred_final'] = y_pred
            
            # 테스트 데이터의 원래 인덱스를 저장 (원래 데이터프레임에서 정보를 가져오기 위함)
            st.session_state['X_test_indices'] = X_test.index if hasattr(X_test, 'index') else range(len(X_test))
            
            st.success("✅ 모델과 예측 결과가 저장되었습니다. 이제 7번 파트를 실행할 수 있습니다!")
            st.markdown("---")
    # ==========================================
    # 7. 전략 수립 및 결과 시각화 (Segmentation)
    # ==========================================
    st.header("7. 고객 세분화 및 전략 수립 (Strategy & Visualization)")

    if 'y_proba_final' not in st.session_state:
        st.warning("⚠️ 먼저 위 5&6번 단계에서 '모델 학습 및 평가' 버튼을 눌러주세요.")
    else:
        # 데이터 가져오기
        y_proba = st.session_state['y_proba_final']
        y_test = st.session_state['y_test_final']
        
        # 데이터프레임 생성
        res_df = pd.DataFrame({
            'Actual_Default': y_test,
            'Prob_Default': y_proba
        })

        st.subheader("📋 고객 등급 세분화 (Customer Segmentation)")
        
        # 사용자가 기준 설정 가능하게 슬라이더 제공
        c_seg1, c_seg2 = st.columns(2)
        safe_cut = c_seg1.slider("우량 고객 기준 (확률 < X)", 0.0, 0.5, 0.2, 0.05, help="이 확률보다 낮으면 우량 고객으로 분류")
        danger_cut = c_seg2.slider("고위험 고객 기준 (확률 > X)", 0.5, 1.0, 0.6, 0.05, help="이 확률보다 높으면 위험 고객으로 분류")

        # 세분화 로직 함수
        def segment_customer(prob):
            if prob < safe_cut: return "Grade A (우량)"
            elif prob > danger_cut: return "Grade C (고위험)"
            else: return "Grade B (중립)"

        res_df['Segment'] = res_df['Prob_Default'].apply(segment_customer)

        # 1. 세그먼트별 통계 요약
        summary = res_df.groupby('Segment').agg(
            Customer_Count=('Prob_Default', 'count'),
            Avg_Prob=('Prob_Default', 'mean'),
            Actual_Default_Rate=('Actual_Default', 'mean')
        ).reset_index()

        # 전략 텍스트 매핑
        strategy_map = {
            "Grade A (우량)": "✅ 금리 인하, 한도 증액 제안, 교차 판매(Cross-sell) 시도",
            "Grade B (중립)": "⚠️ 정기적인 모니터링, 신규 대출 시 심사 강화",
            "Grade C (고위험)": "🚫 대출 거절/축소, 조기 상환 유도, 집중 관리 대상"
        }
        summary['Action_Plan'] = summary['Segment'].map(strategy_map)

        st.table(summary.style.format({
            "Avg_Prob": "{:.2%}",
            "Actual_Default_Rate": "{:.2%}"
        }))

        # 2. 시각화 대시보드
        st.subheader("📊 전략 시각화 대시보드")
        
        row1_1, row1_2 = st.columns(2)

        # 그래프 1: 세그먼트 비율 (Pie Chart)
        with row1_1:
            st.markdown("**1. 고객 등급별 분포**")
            fig1, ax1 = plt.subplots(figsize=(6, 4))
            segment_counts = res_df['Segment'].value_counts().sort_index()
            # 색상 설정 (우량:파랑, 중립:회색, 위험:빨강)
            colors = {'Grade A (우량)': '#66b3ff', 'Grade B (중립)': '#999999', 'Grade C (고위험)': '#ff9999'}
            col_list = [colors.get(x, '#cccccc') for x in segment_counts.index]
            
            ax1.pie(segment_counts, labels=segment_counts.index, autopct='%1.1f%%', colors=col_list, startangle=90)
            st.pyplot(fig1)

        # 그래프 2: 리스크 등급별 부도 확률 분포 (Box Plot)
        with row1_2:
            st.markdown("**2. 등급별 예측 부실률 분포**")
            fig2, ax2 = plt.subplots(figsize=(6, 4))
            sns.boxplot(x='Segment', y='Prob_Default', data=res_df.sort_values('Segment'), palette="Set2", ax=ax2)
            ax2.set_ylabel("Predicted Probability")
            st.pyplot(fig2)

        st.markdown("---")
