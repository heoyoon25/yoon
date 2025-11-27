import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import graphviz # Decision Tree 시각화를 위해 추가
from sklearn.tree import DecisionTreeClassifier, export_graphviz # export_graphviz 추가
from sklearn.metrics import confusion_matrix # Confusion Matrix 계산을 위해 추가

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.impute import SimpleImputer

# 모델 및 성능 평가 라이브러리
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, roc_curve, auc

# 1. 초기 설정 및 세션 상태 관리
st.set_page_config(layout="wide", page_title="이진 분류 분석 웹 애플리케이션")

# 세션 상태 초기화: 데이터와 모델 결과를 저장하여 탭 간에 공유
if 'df' not in st.session_state:
    st.session_state.df = None
if 'processed_df' not in st.session_state:
    st.session_state.processed_df = None
if 'X_train' not in st.session_state:
    st.session_state.X_train = None
if 'results' not in st.session_state:
    st.session_state.results = {}
if 'models' not in st.session_state:
    st.session_state.models = {}

# 2. 애플리케이션 제목 및 메뉴 설정
st.title("이진 분류 데이터 분석 웹 애플리케이션 💻")

# 탭 메뉴 설정
tab_names = ["1. 데이터 업로드", "2. 데이터 탐색 및 시각화", "3. 데이터 전처리", "4. 모델 학습", "5. 성능 평가 및 비교"]
tab1, tab2, tab3, tab4, tab5 = st.tabs(tab_names)


# (tab1, tab2, tab3 코드는 이전과 동일하게 유지됩니다. 생략)
# --- 섹션 1: 데이터 업로드 ---
with tab1:
    st.header("1. 데이터 업로드 📥")
    uploaded_file = st.file_uploader("CSV 파일을 업로드하세요:", type=["csv"])

    if uploaded_file is not None:
        try:
            # low_memory=False로 데이터 유형 추론 오류 방지
            df = pd.read_csv(uploaded_file, low_memory=False)
            st.session_state.df = df
            st.session_state.processed_df = df.copy() # 전처리 시작점으로 사용
            st.success("데이터 로드 성공! '데이터 탐색' 탭으로 이동하세요.")
            st.dataframe(df.head())
        except Exception as e:
            st.error(f"파일을 읽는 중 오류가 발생했습니다: {e}")

# --- 섹션 2: 데이터 탐색 및 시각화 ---
with tab2:
    st.header("2. 데이터 탐색 및 시각화 🔎")

    if st.session_state.df is not None:
        st.subheader("데이터 개요")
        # 데이터 총 개수 및 행/열 개수
        st.write(f"**총 데이터 개수:** {st.session_state.df.size}")
        st.write(f"**행(Row) 개수:** {st.session_state.df.shape[0]}")
        st.write(f"**열(Column, 변수) 개수:** {st.session_state.df.shape[1]}")
        st.write("---")

        st.subheader("시각화 도구")
        
        cols = st.session_state.df.columns.tolist()
        
        col1, col2 = st.columns(2)
        
        with col1:
            chart_type = st.selectbox("그래프 형태 선택:", 
                                      ['산점도(scatter)', '막대 그래프(bar)', '히스토그램(histogram)'])
            
        with col2:
            # 시각화할 변수 선택 (Y_Label은 산점도나 막대 그래프에서만 필요)
            x_label = st.selectbox("X축 변수 선택 (X_Label):", cols)
            y_label_options = [""] + cols
            y_label = st.selectbox("Y축 변수 선택 (Y_Label, 선택 사항):", y_label_options)


        if chart_type and x_label:
            try:
                if chart_type == '산점도(scatter)' and y_label:
                    fig = px.scatter(st.session_state.df, x=x_label, y=y_label, title=f"{x_label} vs {y_label} 산점도")
                elif chart_type == '막대 그래프(bar)' and y_label:
                    # 막대 그래프는 범주형 변수의 빈도 또는 숫자형 변수의 평균 등을 시각화
                    temp_df = st.session_state.df.groupby(x_label)[y_label].mean().reset_index()
                    fig = px.bar(temp_df, x=x_label, y=y_label, title=f"{x_label}별 {y_label} 평균")
                elif chart_type == '히스토그램(histogram)':
                    fig = px.histogram(st.session_state.df, x=x_label, title=f"{x_label} 분포 히스토그램")
                else:
                    st.warning("산점도나 막대 그래프를 선택했을 경우 Y축 변수를 선택해야 합니다.")
                    fig = None
                
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.warning(f"시각화 오류: 선택한 변수 유형이 그래프에 적합하지 않을 수 있습니다. ({e})")
    else:
        st.info("데이터 업로드 탭에서 파일을 먼저 업로드해주세요.")

# --- 섹션 3: 데이터 전처리 ---
with tab3:
    st.header("3. 데이터 전처리 🛠️")
    
    if st.session_state.processed_df is not None:
        temp_df = st.session_state.processed_df.copy()

        st.subheader("3-1. 결측치 처리 (Missing Value Handling)")
        
        missing_info = temp_df.isnull().sum()
        missing_info = missing_info[missing_info > 0].sort_values(ascending=False)
        st.write(f"**현재 결측치 있는 열: 총 {len(missing_info)}개**")
        
        if not missing_info.empty:
            missing_col = st.selectbox("처리할 결측치 변수 선택:", [''] + missing_info.index.tolist(), key="missing_col_select")
            
            if missing_col:
                impute_method = st.selectbox(f"'{missing_col}'의 처리 방법:", 
                                             ['열 삭제(Drop Column)', '평균 대체(Mean Imputation)', '최빈값 대체(Most Frequent Imputation)'], key="impute_method_select")
                
                if st.button(f"'{missing_col}' 변수 결측치 처리 실행", key="run_impute"):
                    if impute_method == '열 삭제(Drop Column)':
                        temp_df = temp_df.drop(columns=[missing_col])
                        st.success(f"'{missing_col}' 열이 삭제되었습니다.")
                    elif impute_method == '평균 대체(Mean Imputation)':
                        if pd.api.types.is_numeric_dtype(temp_df[missing_col]):
                            temp_df[missing_col] = temp_df[missing_col].fillna(temp_df[missing_col].mean())
                            st.success(f"'{missing_col}'의 결측치가 평균으로 대체되었습니다.")
                        else:
                            st.warning("경고: 평균 대체는 숫자형 변수에만 적용할 수 있습니다.")
                    elif impute_method == '최빈값 대체(Most Frequent Imputation)':
                        imputer = SimpleImputer(strategy='most_frequent')
                        # 최빈값 대체는 숫자형/범주형 모두 가능
                        temp_df[missing_col] = imputer.fit_transform(temp_df[[missing_col]])
                        st.success(f"'{missing_col}'의 결측치가 최빈값으로 대체되었습니다.")
                    
                    st.session_state.processed_df = temp_df
                    st.rerun() # 처리 결과 반영을 위해 재실행
        else:
            st.success("모든 변수에 결측치가 없습니다.")
        
        st.write("---")

        st.subheader("3-2. 범주형 변수 인코딩 (Categorical Encoding)")
        
        object_cols = temp_df.select_dtypes(include='object').columns.tolist()
        st.write(f"**인코딩 대기 중인 범주형 변수:** {object_cols if object_cols else '없음'}")
        
        if object_cols:
            cat_col = st.selectbox("인코딩할 범주형 변수 선택:", [''] + object_cols, key="cat_col_select")
            
            if cat_col:
                encoding_method = st.selectbox(f"'{cat_col}'의 인코딩 방법:", ['원-핫 인코딩(One-Hot Encoding)'], key="encoding_method_select")
                
                if st.button(f"'{cat_col}' 변수 인코딩 실행", key="run_encoding"):
                    if encoding_method == '원-핫 인코딩(One-Hot Encoding)':
                        temp_df = pd.get_dummies(temp_df, columns=[cat_col], drop_first=True)
                        st.success(f"'{cat_col}' 변수가 원-핫 인코딩 되었습니다. ({temp_df.shape[1] - st.session_state.processed_df.shape[1]}개 열 추가)")
                    
                    st.session_state.processed_df = temp_df
                    st.rerun()

        st.write("---")
        
        st.subheader("3-3. 스케일링 (Scaling)")
        
        numeric_cols = temp_df.select_dtypes(include=np.number).columns.tolist()
        
        scale_method = st.selectbox("스케일링 방법 선택:", ['선택 안함', '표준화(StandardScaler)', '정규화(MinMaxScaler)'], key="scale_method_select")
        scale_cols = st.multiselect("스케일링할 변수 선택:", numeric_cols, key="scale_cols_select")

        if scale_method != '선택 안함' and scale_cols:
            if st.button(f"선택된 변수 스케일링 실행 ({scale_method})", key="run_scaling"):
                if scale_method == '표준화(StandardScaler)':
                    scaler = StandardScaler()
                else: # MinMaxScaler
                    scaler = MinMaxScaler()
                
                temp_df[scale_cols] = scaler.fit_transform(temp_df[scale_cols])
                st.success(f"선택된 {len(scale_cols)}개 변수가 {scale_method}되었습니다.")
                
                st.session_state.processed_df = temp_df
                st.rerun()
                
        
        st.write("---")
        st.subheader("현재 전처리 상태 미리보기")
        st.dataframe(st.session_state.processed_df.head())
        st.write(f"현재 열 개수: {st.session_state.processed_df.shape[1]}")
    else:
        st.info("데이터 업로드 탭에서 파일을 먼저 업로드해주세요.")


# --- 섹션 4: 모델 학습 (하이퍼파라미터 선택 옵션 강화) ---
with tab4:
    st.header("4. 모델 학습 🧠")
    
    if st.session_state.processed_df is not None:
        
        # 4-1. 종속/독립 변수 설정 및 파티셔닝
        st.subheader("4-1. 종속 변수 (Y) 설정 및 데이터 파티셔닝")
        
        # 숫자형 변수만 선택 가능하도록 제한 (이진 분류 타겟)
        target_cols = st.session_state.processed_df.select_dtypes(include=np.number).columns.tolist()
        target_col = st.selectbox("종속 변수 (Y, 타겟) 선택 (0 또는 1로 분류된 변수):", target_cols, key="target_col_select")
        
        if target_col:
            
            col_split1, col_split2 = st.columns(2)
            with col_split1:
                test_size = st.slider("테스트 데이터 비율 (Test Size):", 0.1, 0.5, 0.3, 0.05)
            with col_split2:
                random_state = st.number_input("랜덤 시드 (Random State):", 0, 100, 42)
            
            # 독립 변수 (X) 설정: 타겟 변수를 제외한 모든 숫자형 변수를 기본으로 사용
            feature_cols = [col for col in target_cols if col != target_col]
            selected_features = st.multiselect("독립 변수 (X, 특징) 선택:", feature_cols, default=feature_cols, key="selected_features_select")

            if not selected_features:
                st.warning("독립 변수를 1개 이상 선택해야 모델 학습이 가능합니다.")

            # 데이터 분할 및 저장
            if st.button("데이터 파티셔닝 실행", key="run_split"):
                # 실제 데이터에 0과 1만 있는지 확인하는 로직 추가
                unique_values = st.session_state.processed_df[target_col].unique()
                if not (np.all(np.isin(unique_values, [0, 1])) and len(unique_values) == 2):
                    st.error("선택된 종속 변수는 오직 0과 1만을 포함하는 이진 분류 변수여야 합니다. 다시 확인해주세요.")
                else:
                    # 선택된 독립 변수와 종속 변수만 사용 (독립 변수 잔여 NaN은 0으로 임시 대체)
                    X = st.session_state.processed_df[selected_features].fillna(0) 
                    y = st.session_state.processed_df[target_col]
                    
                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, test_size=test_size, random_state=random_state, stratify=y
                    )
                    
                    st.session_state.X_train = X_train
                    st.session_state.X_test = X_test
                    st.session_state.y_train = y_train
                    st.session_state.y_test = y_test
                    
                    st.success("데이터 파티셔닝 완료! 모델 학습을 진행하세요.")
                    st.write(f"훈련 데이터 개수: {len(X_train)}")
                    st.write(f"테스트 데이터 개수: {len(X_test)}")
            
            st.write("---")
            
            # 4-2. 모델별 설정 및 학습
            if st.session_state.X_train is not None:
                st.subheader("4-2. 모델별 설정 및 학습")

                # 모델 학습 및 평가 함수
                def train_and_evaluate(model_name, model, X_train, y_train, X_test, y_test):
                    try:
                        model.fit(X_train, y_train)
                        y_pred = model.predict(X_test)
                        # 확률 예측 (ROC 곡선 계산용)
                        y_proba = model.predict_proba(X_test)[:, 1] 
                        
                        # Confusion Matrix 계산
                        cm = confusion_matrix(y_test, y_pred)
                        
                        acc = accuracy_score(y_test, y_pred)
                        rec = recall_score(y_test, y_pred, zero_division=0)
                        prec = precision_score(y_test, y_pred, zero_division=0)
                        f1 = f1_score(y_test, y_pred, zero_division=0)
                        fpr, tpr, thresholds = roc_curve(y_test, y_proba)
                        roc_auc = auc(fpr, tpr)
                        
                        st.session_state.results[model_name] = {
                            'Accuracy': acc, 'Recall': rec, 'Precision': prec, 'F1-Score': f1, 
                            'FPR': fpr, 'TPR': tpr, 'AUC': roc_auc, 'y_proba': y_proba,
                            'Confusion Matrix': cm, 'y_pred': y_pred # CM 및 예측값 저장
                        }
                        st.session_state.models[model_name] = model
                        st.success(f"**{model_name}** 학습 완료! 정확도: {acc:.4f}")
                    except Exception as e:
                        st.error(f"{model_name} 학습 중 오류 발생: {e}")
                        
                
                # --- 모델 1: 의사결정나무 ---
                st.markdown("##### 🌲 의사결정나무 (Decision Tree)")
                
                dt_col1, dt_col2, dt_col3 = st.columns(3)
                with dt_col1:
                    max_depth = st.number_input("최대 깊이 (Max Depth):", 1, 30, 5, key="dt_max_depth")
                with dt_col2:
                    min_samples_split = st.number_input("최소 분할 샘플 수 (Min Samples Split):", 2, 50, 2, key="dt_min_samples_split")
                with dt_col3:
                    # Gini와 Entropy 추가 (요청 사항)
                    criterion = st.selectbox("불순도 기준 (Criterion):", ['gini', 'entropy', 'log_loss'], key="dt_criterion")

                dt_col4, dt_col5 = st.columns(2)
                with dt_col4:
                    # Splitter 추가 (요청 사항)
                    splitter = st.selectbox("분할 전략 (Splitter):", ['best', 'random'], key="dt_splitter")
                with dt_col5:
                    min_samples_leaf = st.number_input("최소 리프 샘플 수 (Min Samples Leaf):", 1, 50, 1, key="dt_min_samples_leaf")
                
                if st.button("의사결정나무 학습 실행", key="run_dt"):
                    dt_model = DecisionTreeClassifier(max_depth=max_depth, 
                                                     min_samples_split=min_samples_split, 
                                                     criterion=criterion, # 반영
                                                     splitter=splitter, # 반영
                                                     min_samples_leaf=min_samples_leaf, # 반영
                                                     random_state=random_state)
                    train_and_evaluate("Decision Tree", dt_model, st.session_state.X_train, st.session_state.y_train, st.session_state.X_test, st.session_state.y_test)

                st.write("---")
                
                # --- 모델 2: Logit (Logistic Regression) ---
                st.markdown("##### 📈 로지스틱 회귀 (Logit)")
                
                logit_col1, logit_col2 = st.columns(2)
                with logit_col1:
                    logit_C = st.slider("규제 강도 (C, 낮을수록 규제 강함):", 0.01, 10.0, 1.0, 0.01, key="logit_C")
                with logit_col2:
                    # Penalty (L1/L2) 추가 (요청 사항)
                    logit_penalty = st.selectbox("규제 종류 (Penalty):", ['l2', 'l1'], key="logit_penalty") 
                
                logit_solver = 'liblinear' if logit_penalty in ['l1', 'l2'] else 'lbfgs' # L1을 위해 liblinear 사용

                if st.button("Logit 학습 실행", key="run_logit"):
                    logit_model = LogisticRegression(C=logit_C, 
                                                     penalty=logit_penalty, # 반영
                                                     solver=logit_solver, 
                                                     random_state=random_state, 
                                                     max_iter=1000)
                    train_and_evaluate("Logit", logit_model, st.session_state.X_train, st.session_state.y_train, st.session_state.X_test, st.session_state.y_test)
                
                st.write("---")
                
                # --- 모델 3: Hybrid (앙상블) ---
                st.markdown("##### 융합 모델 (Hybrid: DT + Logit)")
                
                # Hybrid 모델 학습 가능 여부 확인
                if "Decision Tree" in st.session_state.results and "Logit" in st.session_state.results:
                    
                    st.write("Hybrid 모델은 두 모델의 예측 확률을 가중 평균하여 결과를 도출합니다.")
                    
                    col_w1, col_w2 = st.columns(2)
                    with col_w1:
                        weight_dt = st.slider("Decision Tree 가중치:", 0.0, 1.0, 0.5, 0.05, key="hybrid_w_dt")
                    with col_w2:
                        # Logit 가중치는 1 - weight_dt로 자동 설정
                        weight_logit = 1 - weight_dt
                        st.metric("Logit 가중치 (자동 설정):", f"{weight_logit:.2f}")
                    
                    if st.button("Hybrid 모델 평가 실행", key="run_hybrid"):
                        
                        dt_proba = st.session_state.results["Decision Tree"]['y_proba']
                        logit_proba = st.session_state.results["Logit"]['y_proba']
                        y_test = st.session_state.y_test
                        
                        # 가중 평균 확률 계산
                        hybrid_proba = (dt_proba * weight_dt) + (logit_proba * weight_logit)
                        
                        # 0.5를 기준으로 예측 클래스 결정
                        hybrid_pred = (hybrid_proba >= 0.5).astype(int)
                        
                        # 성능 평가
                        cm = confusion_matrix(y_test, hybrid_pred) # CM 추가
                        acc = accuracy_score(y_test, hybrid_pred)
                        rec = recall_score(y_test, hybrid_pred, zero_division=0)
                        prec = precision_score(y_test, hybrid_pred, zero_division=0)
                        f1 = f1_score(y_test, hybrid_pred, zero_division=0)
                        fpr, tpr, thresholds = roc_curve(y_test, hybrid_proba)
                        roc_auc = auc(fpr, tpr)
                        
                        st.session_state.results["Hybrid"] = {
                            'Accuracy': acc, 'Recall': rec, 'Precision': prec, 'F1-Score': f1, 
                            'FPR': fpr, 'TPR': tpr, 'AUC': roc_auc, 'y_proba': hybrid_proba,
                            'Confusion Matrix': cm, 'y_pred': hybrid_pred # CM 및 예측값 저장
                        }
                        st.success(f"Hybrid 모델 평가 완료! 정확도: {acc:.4f}")

                else:
                    st.warning("Hybrid 모델을 평가하려면 Decision Tree와 Logit 모델을 먼저 학습시키세요.")
            
    else:
        st.info("데이터 업로드 및 전처리 탭을 완료하고 '데이터 파티셔닝 실행' 버튼을 눌러야 모델 학습이 가능합니다.")


# --- 섹션 5: 성능 평가 및 비교 (섹션 순서 변경 및 시각화 추가) ---
with tab5:
    st.header("5. 성능 평가 및 비교 🏆")
    
    if st.session_state.results:
        
        # --- 5-1. 의사결정나무 시각화 --- (기존 5-3)
        st.subheader("5-1. 🌲 Decision Tree 모델 구조 시각화")
        
        if "Decision Tree" in st.session_state.models:
            dt_model = st.session_state.models["Decision Tree"]
            if st.session_state.X_train is not None:
                feature_names = st.session_state.X_train.columns.tolist()
                target_names = [str(int(c)) for c in st.session_state.y_train.unique()]
            else:
                feature_names = None
                target_names = ['0', '1']

            try:
                # Graphviz DOT 데이터를 생성
                dot_data = export_graphviz(dt_model, 
                                           out_file=None, 
                                           feature_names=feature_names,  
                                           class_names=target_names,  
                                           filled=True, 
                                           rounded=True, 
                                           special_characters=True)

                # Streamlit에 Graphviz 이미지 표시
                st.graphviz_chart(dot_data)
                
                st.info("Decision Tree 구조 시각화. (복잡할 경우, 모델 학습 탭에서 Max Depth를 낮춰보세요.)")
            except Exception as e:
                st.warning(f"Decision Tree 시각화 중 오류 발생: {e}. 'graphviz' 라이브러리가 설치되어 있는지 확인해주세요.")
                
        else:
            st.info("Decision Tree 모델을 먼저 학습시키세요.")
            
        st.write("---")
        
        # --- 5-2. Logit 계수 시각화 --- (신규 추가)
        st.subheader("5-2. 📈 Logit (로지스틱 회귀) 모델 계수 시각화")

        if "Logit" in st.session_state.models:
            logit_model = st.session_state.models["Logit"]
            feature_names = st.session_state.X_train.columns.tolist()
            
            # Logit은 하나의 계수 배열만 가짐 (이진 분류)
            coefficients = logit_model.coef_[0] 
            
            coef_df = pd.DataFrame({
                'Feature': feature_names,
                'Coefficient': coefficients
            }).sort_values(by='Coefficient', ascending=True) # 절대값이 아닌 값 자체로 정렬
            
            # 계수 막대 그래프 시각화
            fig_coef = px.bar(coef_df, 
                              x='Coefficient', 
                              y='Feature', 
                              orientation='h', 
                              title='Logit Model Coefficients (계수)',
                              height=max(400, len(feature_names) * 30))
            
            # y축 순서를 계수 크기 순으로 설정
            fig_coef.update_layout(yaxis={'categoryorder':'total ascending'}) 
            st.plotly_chart(fig_coef, use_container_width=True)
            
            st.info("계수(Coefficient)의 크기는 영향력을 나타냅니다. 양수(+)는 타겟 클래스(1)의 확률을 높이고, 음수(-)는 낮춥니다.")
            
        else:
            st.info("Logit 모델을 먼저 학습시키세요.")
            
        st.write("---")
        
        # --- 5-3. 성능 지표 비교표 --- (기존 5-1 -> 5-3)
        st.subheader("5-3. 성능 지표 비교 (Accuracy, Recall, Precision, F1-Score, AUC)")
        
        comparison_data = {}
        for model_name, metrics in st.session_state.results.items():
            comparison_data[model_name] = {
                'Accuracy': f"{metrics['Accuracy']:.4f}",
                'Recall': f"{metrics['Recall']:.4f}",
                'Precision': f"{metrics['Precision']:.4f}",
                'F1-Score': f"{metrics['F1-Score']:.4f}",
                'AUC (ROC 영역)': f"{metrics['AUC']:.4f}"
            }
        
        df_comparison = pd.DataFrame.from_dict(comparison_data, orient='index')
        df_comparison.index.name = "모델"
        st.dataframe(df_comparison)
        
        st.write("---")

        # --- 5-4. ROC 곡선 비교 --- (기존 5-2 -> 5-4)
        st.subheader("5-4. ROC 곡선 비교")
        
        fig_roc = go.Figure()
        
        # 무작위 추측 선 (Random Guess)
        fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1],
                                     mode='lines',
                                     line=dict(dash='dash', color='gray'),
                                     name='Random Guess (AUC = 0.50)'))
                                     
        # 각 모델별 ROC 곡선 추가
        for model_name, metrics in st.session_state.results.items():
            fig_roc.add_trace(go.Scatter(x=metrics['FPR'], y=metrics['TPR'],
                                         mode='lines',
                                         name=f'{model_name} (AUC = {metrics["AUC"]:.4f})'))
        
        fig_roc.update_layout(
            title='ROC Curve Comparison',
            xaxis_title='False Positive Rate (FPR)',
            yaxis_title='True Positive Rate (TPR) / Recall',
            xaxis=dict(range=[0, 1]),
            yaxis=dict(range=[0, 1.05]),
            width=800,
            height=600
        )
        
        st.plotly_chart(fig_roc, use_container_width=True)
        
        st.write("---")

        # --- 5-5. Confusion Matrix (혼동 행렬) 시각화 --- (신규 추가, 기존 5-3 -> 5-5)
        st.subheader("5-5. Confusion Matrix (혼동 행렬)")

        selected_model_cm = st.selectbox("Confusion Matrix를 확인할 모델 선택:", 
                                         list(st.session_state.results.keys()), 
                                         key="cm_model_select")
        
        if selected_model_cm and 'Confusion Matrix' in st.session_state.results[selected_model_cm]:
            cm = st.session_state.results[selected_model_cm]['Confusion Matrix']
            
            # Confusion Matrix 시각화를 위한 DataFrame 생성
            cm_labels = ['Negative (0)', 'Positive (1)']
            cm_df = pd.DataFrame(cm, index=[f'Actual {label}' for label in cm_labels], 
                                 columns=[f'Predicted {label}' for label in cm_labels])

            # Plotly Heatmap 생성
            fig_cm = px.imshow(cm, 
                               text_auto=True, 
                               aspect="auto",
                               color_continuous_scale='Blues',
                               labels=dict(x="Predicted Class", y="Actual Class", color="Count"),
                               x=cm_labels,
                               y=cm_labels,
                               title=f'{selected_model_cm} Confusion Matrix')
            
            # 레이아웃 조정
            fig_cm.update_xaxes(side="top")
            fig_cm.update_layout(
                xaxis_title='Predicted Class', 
                yaxis_title='Actual Class',
                width=600,
                height=600
            )
            
            st.plotly_chart(fig_cm, use_container_width=False)
            
            st.markdown(f"""
            - **True Negative (TN):** {cm[0, 0]} (실제 0, 예측 0)
            - **False Positive (FP):** {cm[0, 1]} (실제 0, 예측 1 - Type I Error)
            - **False Negative (FN):** {cm[1, 0]} (실제 1, 예측 0 - Type II Error)
            - **True Positive (TP):** {cm[1, 1]} (실제 1, 예측 1)
            """)

        else:
            st.info("선택된 모델의 학습 결과를 찾을 수 없거나 Confusion Matrix가 계산되지 않았습니다.")

    else:
        st.info("모델 학습 탭에서 모델을 학습시키고 평가를 진행해야 결과를 볼 수 있습니다.")
