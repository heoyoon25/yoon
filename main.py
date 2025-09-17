import streamlit as st
st.title("연산")
st.header("두 숫자 더하기")

num_a = st.number_input("숫자 A를 입력하세요", value=0, step=1)

num_b = st.number_input("숫자 B를 입력하세요", value=0, step=1)

sum_result = num_a + num_b

st.write(f"두 수의 합은: {sum_result}")

st.markdown('---')

st.header('선택된 숫자까지의 합 구하기')

def format_number(x):
    return f"숫자: {x}"

selected_num = st.selectbox('1부터 더할 숫자를 선택하세요.', options=[num_a, num_b], format_func=format_number)

total = 0
for i in range(1, selected_num + 1):
    total += i

st.write(f"선택한 숫자 {selected_num}에 대한 1부터의 합계는 {total}입니다.")

