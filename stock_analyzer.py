import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
import altair as alt
import plotly.graph_objects as go
import plotly.express as px
import seaborn as sns 
import matplotlib.pyplot as plt
import numpy as np
from assets.functions import *
from assets.alpaca_setup import *

# Set Page Name and Emoji ####################################################################################
st.set_page_config(page_title='Stock Analyzer',layout='wide',page_icon='📈')

# Set Up Sidebar ####################################################################################
with st.sidebar:
    st.subheader('Stock Selection')

    # Load Index facts from Wiki ##########################################################
    url_dict = {'S&P500':{'url':'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies','position':0,'etf':'SPY'},
                'S&P400':{'url':'https://en.wikipedia.org/wiki/List_of_S%26P_400_companies','position':0,'etf':'SPMD'},
                'S&P600':{'url':'https://en.wikipedia.org/wiki/List_of_S%26P_600_companies','position':0,'etf':'SPSM'},
                'DJIA30':{'url':'https://en.wikipedia.org/wiki/Dow_Jones_Industrial_Average#Components','position':2,'etf':'DIA'},
                'Nasdaq100':{'url':'https://en.wikipedia.org/wiki/Nasdaq-100#Components','position':4,'etf':'QQQ'}
                }

    # A function to pull those facts ##########################################################
    def load_data(url_dict):
        df = pd.DataFrame([])
        for key in url_dict:
            html = pd.read_html(url_dict[key]['url'], header=0)
            df1 = html[url_dict[key]['position']]
            df1['Index'] = key
            df1.rename(columns={'Security':'Company'},inplace=True)
            df1.rename(columns={'Ticker':'Symbol'},inplace=True)
            df = pd.concat([df, df1], axis=0, ignore_index=True)
        return df
    
    index_list = list(url_dict.keys())
    df_tickers = load_data(url_dict)
    key_cols = ['Index','Symbol','Company','Headquarters Location','Date added','Founded']
    etf_list = [url_dict[key]['etf'] for key in url_dict]
    symbol_list = list(df_tickers[df_tickers['Index'].isin(index_list)]['Symbol'].unique()) + etf_list
    selected_stocks_list = st.multiselect('Select Stocks ', symbol_list, default=etf_list)

    # Select input parameters
    frequency = st.radio('Select Stock Bar Frequency',['Day','Hour','Minute'],index=0,horizontal=True)
    # Set time unit for Altair chart
    if frequency == 'Day': timeUnit = 'yearmonthdate'
    elif frequency == 'Hour': timeUnit = 'yearmonthdatehours'
    elif frequency == 'Minute': timeUnit = 'yearmonthdatehoursminutes'
    col1, col2 = st.columns(2)
    with col1: date_start = st.date_input("Select Start Date", datetime.today()-timedelta(days=365), format="YYYY/MM/DD") # .replace(month=1, day=1)
    with col2: date_end = st.date_input("Select End Date", datetime.today(), format="YYYY/MM/DD")
    
    # Pull stock data from Alpaca based on user input
    data = get_stock_data_daily(selected_stocks_list, date_start, date_end, frequency=frequency)
    st.divider()

    # Show selected stocks details
    st.subheader('Selected Stocks')
    st.dataframe(df_tickers[df_tickers['Symbol'].isin(selected_stocks_list)][['Index','Symbol','Company']], hide_index=True, use_container_width=True)


# Set Up Multiple Tabs ####################################################################################
tab1, tab2, tab3, tab4 = st.tabs(['🌎 Market Snapshot', '📈 Index Correlation', '📊 Stock Correlation','📂 Reference Data']) # '📋 Stock Data',🎲
# Tweak font size of tab names
font_css = """
<style>
button[data-baseweb="tab"] > div[data-testid="stMarkdownContainer"] > p {
  font-size: 18px;
}
</style>
"""
st.write(font_css, unsafe_allow_html=True)
# Set Up Multiple Tabs ####################################################################################

# Tab 1: Market Condition ####################################################################################
with tab1:
    # Pull data from Alpaca ####################################################################################
    st.subheader('Index Price - Daily Close')
    
    # User Inputs
    col1, col2, col3, col4 = st.columns([1,1,1,4])
    with col1:
        market_date_start = st.date_input("Start Date", datetime.today()-timedelta(days=365), format="YYYY/MM/DD") # .replace(month=1, day=1)
    with col2:
        market_date_end = st.date_input("End Date", datetime.today(), format="YYYY/MM/DD")
    with col3:
        line_sticks_select = st.radio('Select Price Chart Type',['Line','Candlesticks'],index=0)

    # Pull market data
    market_request_params = StockBarsRequest(symbol_or_symbols=etf_list, timeframe=TimeFrame.Day, start=market_date_start, end=market_date_end, adjustment='split')
    market_bars = client.get_stock_bars(market_request_params)
    market_data = market_bars.df.reset_index(level=0)

    # Price charts
    col1, col2, col3 = st.columns(3)
    with col1:
        symbol1 = 'SPY'
        make_ohlc_chart(df=market_data[market_data['symbol']==symbol1],title=f'{symbol1} - Price',line_sticks=line_sticks_select)
    with col2:
        symbol2 = 'DIA'
        make_ohlc_chart(df=market_data[market_data['symbol']==symbol2],title=f'{symbol2} - Price',line_sticks=line_sticks_select)
    with col3:
        symbol3 = 'QQQ'
        make_ohlc_chart(df=market_data[market_data['symbol']==symbol3],title=f'{symbol3} - Price',line_sticks=line_sticks_select)
    st.divider()

    # Asset Correlation Analysis
    st.subheader('Asset Correlation - Daily Price Movement')
    col1, col2, col3, col4 = st.columns([1,1,1,4])
    with col1:
        market_corr_rolling_window = st.number_input(f'Rolling Correlation Days', min_value=1, max_value=120, value=21, step=1)

    # Asset Correlation charts
    col1, col2, col3 = st.columns(3)
    with col1:
        list1 = df_tickers[df_tickers['Index']==get_key_from_value(url_dict,symbol1,'etf')]['Symbol'].to_list()
        calculate_plot_rolling_correlation_benchmark(etf_ticker=symbol1, stock_ticker_list=list1, market_date_start=market_date_start, market_date_end=market_date_end, market_corr_rolling_window=market_corr_rolling_window)
    with col2:
        list2 = df_tickers[df_tickers['Index']==get_key_from_value(url_dict,symbol2,'etf')]['Symbol'].to_list()
        calculate_plot_rolling_correlation_benchmark(etf_ticker=symbol2, stock_ticker_list=list2, market_date_start=market_date_start, market_date_end=market_date_end, market_corr_rolling_window=market_corr_rolling_window)        
    with col3:
        list3 = df_tickers[df_tickers['Index']==get_key_from_value(url_dict,symbol3,'etf')]['Symbol'].to_list()
        calculate_plot_rolling_correlation_benchmark(etf_ticker=symbol3, stock_ticker_list=list3, market_date_start=market_date_start, market_date_end=market_date_end, market_corr_rolling_window=market_corr_rolling_window)
    st.divider()
# Tab 1: Market Condition ####################################################################################

# Tab 4: Reference stocks ####################################################################################
with tab4:
    st.subheader('Reference Data')
    
    # Display stock population in an expander
    stock_population = st.expander('Show Stock Population')
    stock_population.write(f'Displaying {len(df_tickers):,} companies.')
    stock_population.dataframe(df_tickers[key_cols],use_container_width=True,hide_index=True)
    
    # Display Sample Data in an expander
    sample_data = st.expander('Show Underlying Bar Data')
    sample_data.dataframe(data,use_container_width=True)
# Tab 4: Reference stocks ####################################################################################

# Tab 2: Index Correlation  ####################################################################################
with tab2:
    tab2_rolling_day = st.number_input(f'Rolling Day Window', min_value=1, max_value=120, value=21, step=1)
    show_top_n = 5
    col1, col2, col3 = st.columns(3)
    with col1:
        ################################# SPY #################################
        st.subheader(f'{symbol1} - Constituent Correlation')
        df_rolling_corr1 = calculate_etf_constituent_corr(etf_ticker=symbol1, stock_ticker_list=list1, start=market_date_start, end=market_date_end, window_size=tab2_rolling_day)
        col1a, col1b = st.columns(2)
        with col1a:
            st.write(f'Top {show_top_n}')
            st.dataframe(df_rolling_corr1.mean(axis=0).sort_values(ascending=False)[:show_top_n])
            top1 = df_rolling_corr1.mean(axis=0).sort_values(ascending=False).index[0]
        with col1b:
            st.write(f'Bottom {show_top_n}')
            st.dataframe(df_rolling_corr1.mean(axis=0).sort_values(ascending=True)[:show_top_n])
            bottom1 = df_rolling_corr1.mean(axis=0).sort_values(ascending=True).index[0]
        st.divider()
        
        selected_stocks1 = st.multiselect('Choose subject stocks',options=df_rolling_corr1.columns.unique(),default=[top1,bottom1])
        df_rolling_corr1_unstack = df_rolling_corr1.unstack().reset_index()
        chart1 = alt.Chart(df_rolling_corr1_unstack[df_rolling_corr1_unstack['symbol'].isin(selected_stocks1)],title=f'Correlation vs. {symbol1}').mark_line().encode(
            x=alt.X('timestamp:O', timeUnit='yearmonthdate', title='Date'), 
            y=alt.Y('0:Q', scale=alt.Scale(zero=False), title=None),
            color=alt.Color('symbol:N',legend=alt.Legend(title=' ',orient='top')))
        st.altair_chart(chart1, use_container_width=True)


    with col2:
        ################################# DIA #################################
        st.subheader(f'{symbol2} - Constituent Correlation')
        df_rolling_corr2 = calculate_etf_constituent_corr(etf_ticker=symbol2, stock_ticker_list=list2, start=market_date_start, end=market_date_end, window_size=tab2_rolling_day)
        col2a, col2b = st.columns(2)
        with col2a:
            st.write(f'Top {show_top_n}')
            st.dataframe(df_rolling_corr2.mean(axis=0).sort_values(ascending=False)[:show_top_n])
            top2 = df_rolling_corr2.mean(axis=0).sort_values(ascending=False).index[0]
        with col2b:
            st.write(f'Bottom {show_top_n}')
            st.dataframe(df_rolling_corr2.mean(axis=0).sort_values(ascending=True)[:show_top_n])
            bottom2 = df_rolling_corr2.mean(axis=0).sort_values(ascending=True).index[0]
        st.divider()

        selected_stocks2 = st.multiselect('Choose subject stocks',options=df_rolling_corr2.columns.unique(),default=[top2,bottom2])
        df_rolling_corr2_unstack = df_rolling_corr2.unstack().reset_index()
        chart2 = alt.Chart(df_rolling_corr2_unstack[df_rolling_corr2_unstack['symbol'].isin(selected_stocks2)],title=f'Correlation vs. {symbol2}').mark_line().encode(
            x=alt.X('timestamp:O', timeUnit='yearmonthdate', title='Date'), 
            y=alt.Y('0:Q', scale=alt.Scale(zero=False), title=None),
            color=alt.Color('symbol:N',legend=alt.Legend(title=' ',orient='top')))
        st.altair_chart(chart2, use_container_width=True)


    with col3:
        ################################# QQQ #################################
        st.subheader(f'{symbol3} - Constituent Correlation')
        df_rolling_corr3 = calculate_etf_constituent_corr(etf_ticker=symbol3, stock_ticker_list=list3, start=market_date_start, end=market_date_end, window_size=tab2_rolling_day)
        col3a, col3b = st.columns(2)
        with col3a:
            st.write('Top 10')
            st.dataframe(df_rolling_corr3.mean(axis=0).sort_values(ascending=False)[:show_top_n])
            top3 = df_rolling_corr3.mean(axis=0).sort_values(ascending=False).index[0]
        with col3b:
            st.write('Bottom 10')
            st.dataframe(df_rolling_corr3.mean(axis=0).sort_values(ascending=True)[:show_top_n])
            bottom3 = df_rolling_corr3.mean(axis=0).sort_values(ascending=True).index[0]
        st.divider()

        selected_stocks3 = st.multiselect('Choose subject stocks',options=df_rolling_corr3.columns.unique(),default=[top3,bottom3])
        df_rolling_corr3_unstack = df_rolling_corr3.unstack().reset_index()
        chart3 = alt.Chart(df_rolling_corr3_unstack[df_rolling_corr3_unstack['symbol'].isin(selected_stocks3)],title=f'Correlation vs. {symbol3}').mark_line().encode(
            x=alt.X('timestamp:O', timeUnit='yearmonthdate', title='Date'), 
            y=alt.Y('0:Q', scale=alt.Scale(zero=False), title=None),
            color=alt.Color('symbol:N',legend=alt.Legend(title=' ',orient='top')))
        st.altair_chart(chart3, use_container_width=True)
# Tab 2: Index Correlation  ####################################################################################

# Tab 3: Correlation Analysis ####################################################################################
with tab3:
    st.subheader('Stock Correlation Analysis')
    df = data.copy()
    df_return = get_stock_return_daily(df)
    # make 2 columns
    col1, col2, col3 = st.columns([4, 0.1, 6])
    # Clustermap ####################################################################################
    with col1:
        # Cluster heatmap of all chosen stocks
        st.subheader(f'Asset Clustermap: {date_start:%Y/%m/%d} - {date_end:%Y/%m/%d}')
        cm = calculate_plot_clustermap(df_return,figsize=(10,10))
        st.pyplot(cm, bbox_inches='tight', dpi=300)
    # Clustermap ####################################################################################
    with col3:
        ################ Pair Correlation ################################################################
        st.subheader('Pair correlation')
        # select a specific pair
        col1, col2, col3 = st.columns(3)
        with col1:
            selected1 = st.selectbox('Select Stock 1', selected_stocks_list, index=0)
            index1 = get_key_from_value(url_dict, selected1, 'etf')
        with col2: 
            selected2 = st.selectbox('Select Stock 2', [x for x in selected_stocks_list if x != selected1] , index=0)
            index2 = get_key_from_value(url_dict, selected2, 'etf')
        with col3: 
            if frequency == 'Day': max_val = 120
            elif frequency == 'Hour': max_val = 24
            elif frequency == 'Minute': max_val = 60*24
            rolling_window = st.number_input(f'Rolling Window - {frequency}s', min_value=1, max_value=max_val, value=21, step=1)
        
        # plot rolling correlation
        df_return['rolling_corr'] = df_return[selected1].rolling(rolling_window).corr(df_return[selected2])
        
        ################ Altair chart ################################################
        # Encoding Types: Q (quantitative), N (nominal), O (ordinal), T (temporal)
        corr_chart = alt.Chart(df_return.reset_index(),title=f'Rolling {rolling_window} {frequency} correlation - {index1} vs. {index2}').mark_line(interpolate='step-after').encode(
            x=alt.X('timestamp:O', timeUnit=timeUnit, title='Date'),
            y=alt.Y('rolling_corr:Q',  title='Correlation', scale=alt.Scale(domain=[-1, 1], zero=False)),
        )
        lines = alt.Chart(pd.DataFrame({'y': [-1,0,1]})).mark_rule().encode(y='y')
        st.altair_chart(corr_chart+lines, use_container_width=True)
        ################ Altair chart ################################################
        ################ Pair Correlation ################################################################
# Tab 3: Correlation Analysis ####################################################################################
