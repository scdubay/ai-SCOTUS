"""
app.py — navigation entry point.

Defines page order and delegates all rendering to pages/.

Run with:
    streamlit run app.py
"""

import streamlit as st

pg = st.navigation([
    st.Page("pages/the_story.py", title="The Story", icon="📖"),
    st.Page("pages/scotus_app.py", title="SCOTUS Legal Aid", icon="⚖️"),
    st.Page("pages/how_it_works.py", title="How it works", icon="📘"),
])
pg.run()
