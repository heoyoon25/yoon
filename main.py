# ==============================================================================
#  단계 3：모델 학습 (데이터 분할을 모델 설정 섹션의 최상단에 통합)
# ==============================================================================
elif st.session_state.step == 3:
    st.subheader("🚀 모델 학습 설정")
    
    if "X_processed" not in st.session_state.data:
        st.warning("⚠️ 먼저 [데이터 전처리] 단계를 완료하세요.")
    else:
        # -------------------------------------------------------------
        # 1. 분석 유형 선택
        # -------------------------------------------------------------
        st.markdown("### 1️⃣ 분석 유형 선택")
        task_option = st.radio(
            "데이터의 타겟(Y) 특성에 맞는 유형을 선택하세요:",
            ["분류 (Classification) - 예: 합격/불합격, 0/1", 
             "회귀 (Regression) - 예: 가격, 점수, 수치 예측"],
            horizontal=True
        )
        st.session_state.task = "logit" if "분류" in task_option else "tree"
        
        st.divider()

        # -------------------------------------------------------------
        # 2. 모델 설정 및 데이터 분할 (통합됨)
        # -------------------------------------------------------------
        st.markdown("### 2️⃣ 모델 설정 및 데이터 분할")
        
        # [A] 데이터 분할 설정 (3개 모델 공통 적용 - 가장 먼저 설정)
        st.markdown("#### ⚙️ 데이터 분할 (3개 모델 공통)")
        test_size = st.slider(
            "테스트 데이터 비율 (검증용)", 
            0.1, 0.4, 0.2, 
            help="전체 데이터 중 학습에 사용하지 않고 검증용으로 남겨둘 데이터의 비율입니다. 3개 모델 모두 동일하게 적용됩니다."
        )
        
        st.markdown("---")

        # [B] 모델별 상세 설정 (Logic / Tree / Hybrid)
        st.markdown("#### 🛠️ 모델별 상세 설정")
        col1, col2, col3 = st.columns(3)
        
        # [Logic 모델]
        with col1:
            st.markdown("##### 🔹 Logic 모델")
            st.caption("선형/로지스틱 회귀")
            st.info("🔧 **설정**: Standard (기본)")

        # [Tree 모델]
        with col2:
            st.markdown("##### 🌳 Tree 모델")
            st.caption("의사결정나무")
            tree_depth = st.slider("최대 깊이 (Max Depth)", 1, 20, 5, key="tree_depth")
            st.caption(f"깊이 제한: {tree_depth}")

        # [Hybrid 모델]
        with col3:
            st.markdown("##### ⚖️ Hybrid 모델")
            st.caption("Logic + Tree 결합")
            reg_weight = st.slider("Logic 가중치", 0.0, 1.0, 0.5, 0.1, key="reg_weight")
            st.caption(f"비율: Logic {int(reg_weight*100)}% : Tree {int((1-reg_weight)*100)}%")

        st.divider()

        # -------------------------------------------------------------
        # 3. 학습 시작 버튼
        # -------------------------------------------------------------
        if st.button("🏁 모델 학습 시작", type="primary"):
            with st.spinner("3가지 모델을 모두 학습 중입니다..."):
                try:
                    X = st.session_state.data["X_processed"]
                    y = st.session_state.data["y_processed"]
                    
                    # 데이터 분할
                    stratify_opt = y if st.session_state.task == "logit" and y.nunique() > 1 else None
                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, test_size=test_size, random_state=42, stratify=stratify_opt
                    )
                    
                    # 모델 초기화
                    if st.session_state.task == "logit":
                        reg_model = LogisticRegression(max_iter=1000)
                        dt_model = DecisionTreeClassifier(max_depth=tree_depth, random_state=42)
                    else:
                        reg_model = LinearRegression()
                        dt_model = DecisionTreeRegressor(max_depth=tree_depth, random_state=42)
                    
                    # 학습 수행
                    reg_model.fit(X_train, y_train)
                    dt_model.fit(X_train, y_train)
                    
                    # 결과 저장
                    st.session_state.models["regression"] = reg_model
                    st.session_state.models["decision_tree"] = dt_model
                    st.session_state.models["mixed_weights"] = {
                        "regression": reg_weight,
                        "decision_tree": 1.0 - reg_weight
                    }
                    st.session_state.data.update({"X_test": X_test, "y_test": y_test})

                    # 완료 메시지
                    st.success("✅ 모든 모델의 학습이 완료되었습니다!")
                    st.info("👉 **'성능 평가' 단계로 이동하여 3개 모델의 성능을 비교하세요.**")
                    
                except Exception as e:
                    st.error(f"학습 중 오류 발생: {e}")
