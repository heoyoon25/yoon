import streamlit as st
st.title('연산')
st.header('두 숫자 더하기')

num_a = st.number_input("숫자 A를 입력하세요", value=0, step=1)

num_b = st.number_input("숫자 B를 입력하세요", value=0, step=1)

sum_result = num_a + num_b

st.write(f"두 수의 합은: {sum_result}")

st.markdown('---')


