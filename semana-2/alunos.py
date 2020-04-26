import streamlit as st
import pandas as pd

def main ():
    st.title('Hello Word')
    st.header('This is a header')
    st.subheader('This is a subheader')
    st.text('This is a Text')
    st.image('Logo.png')

    st.markdown('Botão')
    botao = st.button('Botão')
    if botao:
        st.markdown('Clicado')

    st.markdown('Checkbox')
    check = st.checkbox('Checkbox')
    if check:
        st.markdown('Clicado')

    st.markdown('Radio')
    radio = st.radio('Escolha as Opções:', ('Option 01', 'Option 02', 'Option 03'))
    if radio == 'Option 01':
        st.markdown('Option 01')
    if radio == 'Option 02':
        st.markdown('Option 02')
    if radio == 'Option 03':
        st.markdown('Option 03')

    st.markdown('Selectbox')
    select = st.selectbox('Chose:', ('Option 01', 'Option 02', 'Option 03'))
    if select == 'Option 01':
        st.markdown('Option 01')
    if select == 'Option 02':
        st.markdown('Option 02')
    if select == 'Option 03':
        st.markdown('Option 03')


    st.markdown('Multi')
    multi = st.multiselect('Chose:', ('Option 01', 'Option 02', 'Option 03'))
    if multi == 'Option 01':
        st.markdown('Option 01')
    if multi == 'Option 02':
        st.markdown('Option 02')
    if multi == 'Option 03':
        st.markdown('Option 03')

    st.markdown('File Uploder')
    file = st.file_uploader('Choshe you file',type='csv')
    if file is not None:
        slider = st.slider('Valores', 1,100)
        df = pd.read_csv(file)
        st.dataframe(df.head(slider))
        st.markdown('Markdown')
        st.table(df.head(slider))
        st.write(df.columns)

if __name__ == '__main__':
    main()
