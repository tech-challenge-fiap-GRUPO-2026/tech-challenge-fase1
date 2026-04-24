
import streamlit as st

class StreamlitHelper(object):

    def write(self, text):
        st.write(text)

    def display_divider(self):
        st.divider()

    def display_title(
        self,
        title,
        help=None,
        width='stretch',
    ):
        st.title(
            title,
            help=help,
            width=width
        )

    def display_header(self,
        text, 
        divider=False
    ):
        st.header(
            text,
            divider=divider
        )

    def display_subheader(self,
        text, 
        divider=False
    ):
        st.subheader(
            text,
            divider=divider
        )
    
    def display_text(self, text):
        st.text(text)

    def display_markdown(
        self, 
        text,
        unsafe_allow_html=False
    ):
        st.markdown(
            text,
            unsafe_allow_html=unsafe_allow_html
        )

    def display_code(
        self,
        text,
        language=None,
        wrap_lines=False
    ):
        st.code(
            text,
            language=language,
            wrap_lines=wrap_lines
        )

    def display_dataframe(self, df):
        st.dataframe(df)

    def display_table(
        self,
        data,
        border=True,
        width='stretch',
        height='content'
    ):
        st.table(
            data,
            border=border,
            width=width,
            height=height
        )

    def display_json(
        self,
        data,
        expanded=True,
        width='stretch',
    ):
        st.json(
            data,
            expanded=expanded,
            width=width
        )

    def display_chart(self, chart):
        st.pyplot(chart)