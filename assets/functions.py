import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
import altair as alt
import plotly.graph_objects as go
import plotly.express as px
import seaborn as sns 
import matplotlib.pyplot as plt
import numpy as np
from assets.alpaca_setup import *

# pull market data from alpaca - cached
@st.cache_data(ttl=3600, show_spinner=True) # max cache age 1 hour (3600 seconds)
def get_market_data_daily(etf_list, market_date_start, market_date_end):
    market_request_params = StockBarsRequest(symbol_or_symbols=etf_list, timeframe=TimeFrame.Day, start=market_date_start, end=market_date_end, adjustment='split')
    market_bars = client.get_stock_bars(market_request_params)
    market_data = market_bars.df.reset_index(level=0)
    return market_data

# pull stock data from alpaca
@st.cache_data(ttl=3600, show_spinner=True) # max cache age 1 hour (3600 seconds)
def get_stock_data_daily(symbol_or_symbols, start, end):
    request_params = StockBarsRequest(symbol_or_symbols=symbol_or_symbols, timeframe=TimeFrame.Day, start=start, end=end, adjustment='split')
    bars = client.get_stock_bars(request_params)
    data = bars.df.reset_index(level=0)
    return data

# pull stock data from alpaca and convert to return format
def get_stock_return_daily(symbol_or_symbols, start, end):
    df = get_stock_data_daily(symbol_or_symbols, start, end)
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
        fig.update_layout(yaxis_title=None)
    elif line_sticks == 'Candlesticks':
        fig = go.Figure(data=[go.Candlestick(x=    df.index,
                                            open=  df['open'],
                                            high=  df['high'],
                                            low=   df['low'],
                                            close= df['close'])])
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

# calculate rolling correlation among stocks
def calculate_rolling_correlation(df_stock_return, window_size):
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
    if not isinstance(df_stock_return.index, pd.DatetimeIndex):
        try:
            df_stock_return.index = pd.to_datetime(df_stock_return.index)
        except:
            raise ValueError("DataFrame index must be a DatetimeIndex or convertible to a datetime.")

    rolling_correlations = df_stock_return.rolling(window=window_size).corr(pairwise=True)

    def calculate_upper_triangle_metrics(corr_matrix):
         # Avoid unnecessary calculations if there is only one column.
        if corr_matrix.shape[0] <= 1:
           return pd.Series([np.nan, np.nan, np.nan], index=['avg_corr', 'avg_abs_corr', 'median_corr'])

        # Create an upper triangle mask (excluding the diagonal).
        mask = np.triu(np.ones(corr_matrix.shape, dtype=bool), k=1)
        masked_corr = np.where(mask, corr_matrix, np.nan)
        
        # Calculate the mean, median and absolute mean of the upper triangle (excluding the diagonal).
        # avg_corr = np.nanmean(masked_corr)
        avg_abs_corr = np.nanmean(np.abs(masked_corr))
        median_corr = np.nanmedian(masked_corr)
        corr_metrics = pd.Series([avg_abs_corr, median_corr], index=['avg_abs_corr', 'median_corr'])
        return corr_metrics

    # Calculate the rolling average of the upper triangle of the correlation matrices
    rolling_correlation_metrics = rolling_correlations.groupby(level=0).apply(calculate_upper_triangle_metrics)

    return rolling_correlation_metrics

# calculate rolling correlation against market index
def calculate_rolling_correlation_benchmark(df_stock_return, benchmark_col, window_size):
    
    rolling_corr_df = df_stock_return.rolling(window=window_size).corr(pairwise=True, other=df_stock_return[benchmark_col])
    rolling_corr_df = rolling_corr_df.drop(columns=[benchmark_col])

    def calculate_correlation_metrics(rolling_corr_df):
        avg_abs_corr = np.nanmean(np.abs(rolling_corr_df))
        median_corr = np.nanmedian(rolling_corr_df)
        corr_metrics = pd.Series([avg_abs_corr, median_corr], index=['avg_abs_corr', 'median_corr'])
        return corr_metrics
    
    metric_df = rolling_corr_df.groupby(level=0).apply(calculate_correlation_metrics)

    return metric_df

# calculate and plot rolling correlation against market index
def calculate_plot_rolling_correlation_benchmark(etf_ticker, stock_ticker_list, market_date_start, market_date_end, market_corr_rolling_window=21, timeUnit='yearmonthdate'):
    df_stock_return = get_stock_return_daily(stock_ticker_list+[etf_ticker], start=market_date_start, end=market_date_end)
    df_corr_metric = calculate_rolling_correlation_benchmark( df_stock_return, etf_ticker, window_size=market_corr_rolling_window
                                                             ).unstack().reset_index().rename(columns={'level_0':'metric',0:'correlation'}).set_index('timestamp')
    # Altair chart
    # Encoding Types: Q (quantitative), N (nominal), O (ordinal), T (temporal)
    corr_chart = alt.Chart(df_corr_metric.reset_index(),title=f'{etf_ticker} - Asset Correlation').mark_line().encode(
        x=alt.X('timestamp:O', timeUnit=timeUnit, title='Date'), 
        y=alt.Y('correlation:Q', scale=alt.Scale(zero=False,domain=[0, 1]), title=None),
        color=alt.Color('metric:N',
                        scale={'range':['red','green']},
                        legend=alt.Legend(title=' ',orient='top')))
    return st.altair_chart(corr_chart, use_container_width=True)

# ARCHIVED since individual correlation pairs are too compute intensive
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

