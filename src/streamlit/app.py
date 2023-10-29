import datetime

import numpy as np
import pandas as pd

import streamlit as st
from streamlit_app import predict_page


def main():
    st.title("This is my test page")
    most_recent_data_df = predict_page.fetch_data()
    predict_page.show_predict_page(most_recent_data_df)

    st.title("This is my test page")
    st.table(most_recent_data_df.head())


if __name__ == "__main__":
    main()
