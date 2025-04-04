import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
import financedatabase as fd
import altair as alt
import plotly.graph_objects as go
import plotly.express as px
import seaborn as sns 
import matplotlib.pyplot as plt
import numpy as np
from assets.alpaca_setup import *

# pull stock data from alpaca
def get_stock_data_daily(client, symbol_or_symbols, start, end):
    request_params = StockBarsRequest(symbol_or_symbols=symbol_or_symbols, timeframe=TimeFrame.Day, start=start, end=end, adjustment='split')
    bars = client.get_stock_bars(request_params)
    return bars.df.reset_index(level=0)

# pull stock data from alpaca and convert to return format
def get_stock_return_daily(client, symbol_or_symbols, start, end):
    df = get_stock_data_daily(client, symbol_or_symbols, start, end)
    return df.pivot_table(index='timestamp',columns='symbol',values='close',aggfunc='mean').sort_index(ascending=True).pct_change()



# reformat label
def change_label_style(label, font_size='12px', font_color='black', font_family='sans-serif'):
    html = f"""
    <script>
        var elems = window.parent.document.querySelectorAll('p');
        var elem = Array.from(elems).find(x => x.innerText == '{label}');
        elem.style.fontSize = '{font_size}';
        elem.style.color = '{font_color}';
        elem.style.fontFamily = '{font_family}';
    </script>
    """
    st.components.v1.html(html)

# lookup dict keys based on value
def get_key_from_value(input_dict, value, value_identifier):
    for key, val in input_dict.items():
        if val[value_identifier] == value:
            return key
    return None

# OHLC chart generator
def make_ohlc_chart(df,title='OHLC Chart',line_sticks='Line'):
    if line_sticks =='Line':
        fig = px.line(df, x=df.index, y='close',title=title)
    elif line_sticks == 'Candlesticks':
        fig = go.Figure(data=[go.Candlestick(x=    df.index,
                                            open=  df['open'],
                                            high=  df['high'],
                                            low=   df['low'],
                                            close= df['close'] )])
        fig.update_layout(title=title,xaxis_rangeslider_visible=False,yaxis_title=None)
    return fig 

# calculate average correlation
def calculate_average_corr(df_corr):
    num_stocks=df_corr.shape[0]
    if num_stocks>1:
        upper_triangle = np.triu(np.ones(df_corr.shape), k=1).astype(bool)
        unique_pairwise_correlations = df_corr.where(upper_triangle)
        avg_corr = np.mean(unique_pairwise_correlations)
        avg_abs_corr = np.mean(np.abs(unique_pairwise_correlations))
        return avg_corr, avg_abs_corr
    else: return 'Need at least 2 stocks to calculate correlation metrics.'

# calculate rolling correlation
def calculate_rolling_correlation(df, window_size):
    """
    Calculates the rolling window average correlation for all column values across a DataFrame,
    including only the upper triangle of the correlation matrix, excluding the diagonal, 
    and calculates average correlation, average absolute correlation, and median correlation across all column pairs.

    Args:
        df (pd.DataFrame): DataFrame with stock data indexed by date.
        window_size (int or str): Rolling window size.

    Returns:
         pd.DataFrame: DataFrame containing the rolling average correlation values.
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        try:
            df.index = pd.to_datetime(df.index)
        except:
            raise ValueError("DataFrame index must be a DatetimeIndex or convertible to a datetime.")

    rolling_correlations = df.rolling(window=window_size).corr(pairwise=True)

    def calculate_upper_triangle_metrics(corr_matrix):
         # Avoid unnecessary calculations if there is only one column.
        if corr_matrix.shape[0] <= 1:
           return pd.Series([np.nan, np.nan, np.nan], index=['avg_corr', 'avg_abs_corr', 'median_corr'])

        # Create an upper triangle mask (excluding the diagonal).
        mask = np.triu(np.ones(corr_matrix.shape, dtype=bool), k=1)
        masked_corr = np.where(mask, corr_matrix, np.nan)
        
        # Calculate the mean, median and absolute mean of the upper triangle (excluding the diagonal).
        avg_corr = np.nanmean(masked_corr)
        avg_abs_corr = np.nanmean(np.abs(masked_corr))
        median_corr = np.nanmedian(masked_corr)
       
        return pd.Series([avg_corr, avg_abs_corr, median_corr], index=['avg_corr', 'avg_abs_corr', 'median_corr'])

    # Calculate the rolling average of the upper triangle of the correlation matrices
    rolling_correlation_metrics = rolling_correlations.groupby(level=0).apply(calculate_upper_triangle_metrics)

    return rolling_correlation_metrics

# Plot rolling correlation in Altair
def plot_rolling_correlation(df_rolling_corr,timeUnit='yearmonthdate',title='Rolling Correlation',add_boundary=False):
    # Encoding Types: Q (quantitative), N (nominal), O (ordinal), T (temporal)
    corr_chart = alt.Chart(df_rolling_corr.reset_index(),title=title).mark_line(interpolate='step-after').encode(
        x=alt.X('timestamp:O', timeUnit=timeUnit, title='Date'),
        y=alt.Y('correlation:Q',  title='Correlation', scale=alt.Scale(zero=False)), # domain=[-1, 1], 
        color='metric:N',
    )
    if add_boundary == True:
        lines = alt.Chart(pd.DataFrame({'y': [-1,0,1]})).mark_rule().encode(y='y')
        return st.altair_chart(corr_chart+lines, use_container_width=True)
    else: return st.altair_chart(corr_chart, use_container_width=True)