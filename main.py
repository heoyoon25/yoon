import streamlit as st
st.title('연산')
st.header('두 숫자 더하기')

A = input("숫자 A를 입력하세요: ")

B = input("숫자 B를 입력하세요: ")

sum_result = int(A) + int(B)

print(f"두 수의 합은: {sum_result}")

st.markdown('---')


