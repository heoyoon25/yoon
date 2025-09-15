import streamlit as st
st.title('연산')

st.header('두 숫자 더하기')
col1, col2 = st.columns(2)

with col1:
    A = st.text_input('첫 번째 숫자 (A)를 입력하세요:', key='A_input')
with col2:
    B = st.text_input('두 번째 숫자 (B)를 입력하세요:', key='B_input')
try:
    num_A = int(A)
    num_B = int(B)
    sum_result = num_A + num_B
    st.success(f'두 수의 합은?{sum_result}')

except ValueError:
    st.error('유효한 숫자를 입력해 주세요')


st.markdown('---')
st.header('선택된 숫자까지의 합계 계산')

options = []
if A and A.isdigit():
    options.append(int(A))
if B and B.isdigit():
    options.append(int(B))

if options:
    selected_number = st.selectbox('합계를 구할 숫자를 선택하세요:', options)

    total_sum = selected_number * (selected_number + 1) // 2

    st.info(f'1부터 {selected_number}까지의 모든 정수 합은 {total_sum}입니다.')
else:
    st.warning('합계를 계산하려면 먼저 A와 B에 유효한 정수를 입력해 주세요.')
