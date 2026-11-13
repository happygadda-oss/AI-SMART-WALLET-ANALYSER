"""
styles.py
---------
Small helper for loading external .css files into a Streamlit app.

Streamlit doesn't support linking a .css file the normal HTML way
(<link rel="stylesheet" ...>), so instead we read the file's contents
and inject them into the page wrapped in a <style> tag via st.markdown.

This keeps all our CSS in real .css files (easier to read/edit) instead
of long strings inline inside app.py / auth.py.
"""

import streamlit as st


def load_css(css_file_path: str) -> None:
    """Read a .css file and inject its contents into the current page."""
    with open(css_file_path, "r") as f:
        css = f.read()
    st.markdown(f"<style>\n{css}\n</style>", unsafe_allow_html=True)
