
import streamlit as st

class StreamlitHelper(object):

    def write(self, text):
        st.write(text)

    def display_divider(self):
        st.divider()

    def display_title(self, title):
        st.title(title)

    def display_header(self,
        text, 
        divider=False
    ):
        st.header(text, divider=divider)
    
    def display_text(self, text):
        st.text(text)

    def display_dataframe(self, df):
        st.dataframe(df)

    def display_table(self, data):
        st.table(data)

    def display_chart(self, chart):
        st.pyplot(chart)