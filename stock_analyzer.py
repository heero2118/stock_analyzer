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
            url = url_dict[key]['url']
            html = pd.read_html(url, header=0)
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
    st.divider()

    # Show selected stocks details
    st.subheader('Selected Stocks')
    st.dataframe(df_tickers[df_tickers['Symbol'].isin(selected_stocks_list)][['Index','Symbol','Company']], hide_index=True, use_container_width=True)

# Set Up Multiple Tabs ####################################################################################
tab1, tab2, tab3, tab4 = st.tabs(['🌎 Market Condition', '📂 Reference','📋 Stock Data','📊 Stock Correlation'])
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
    st.subheader('Market Condition')
    # Settings
    col1, col2, col3, col4, col5 = st.columns([1,1,1,1,3])
    with col1:
        market_date_start = st.date_input("Start Date", datetime.today()-timedelta(days=365), format="YYYY/MM/DD") # .replace(month=1, day=1)
    with col2:
        market_date_end = st.date_input("End Date", datetime.today(), format="YYYY/MM/DD")
    with col3:
        line_sticks_select = st.radio('Select Price Chart Type',['Line','Candlesticks'],index=0)
    with col4:
        market_corr_rolling_window = st.number_input(f'Rolling Correlation Days', min_value=1, max_value=120, value=21, step=1)
        
        market_request_params = StockBarsRequest(symbol_or_symbols=etf_list, timeframe=TimeFrame.Day, start=market_date_start, end=market_date_end, adjustment='split')
        market_bars = client.get_stock_bars(market_request_params)
        market_data = market_bars.df.reset_index(level=0)
    # Price charts
    col1, col2, col3 = st.columns(3)
    with col1:
        symbol1 = 'SPY'
        st.plotly_chart(make_ohlc_chart(df=market_data[market_data['symbol']==symbol1],title=symbol1,line_sticks=line_sticks_select), use_container_width=True)
        list1 = df_tickers[df_tickers['Index']==get_key_from_value(url_dict,symbol1,'etf')]['Symbol'].to_list()
        calculate_plot_rolling_correlation_benchmark(etf_ticker=symbol1, stock_ticker_list=list1, market_date_start=market_date_start, market_date_end=market_date_end, market_corr_rolling_window=market_corr_rolling_window)

    with col2:
        symbol2 = 'DIA'
        st.plotly_chart(make_ohlc_chart(df=market_data[market_data['symbol']==symbol2],title=symbol2,line_sticks=line_sticks_select), use_container_width=True)
        list2 = df_tickers[df_tickers['Index']==get_key_from_value(url_dict,symbol2,'etf')]['Symbol'].to_list()
        calculate_plot_rolling_correlation_benchmark(etf_ticker=symbol2, stock_ticker_list=list2, market_date_start=market_date_start, market_date_end=market_date_end, market_corr_rolling_window=market_corr_rolling_window)
        
    with col3:
        symbol3 = 'QQQ'
        st.plotly_chart(make_ohlc_chart(df=market_data[market_data['symbol']==symbol3],title=symbol3,line_sticks=line_sticks_select), use_container_width=True)
        list3 = df_tickers[df_tickers['Index']==get_key_from_value(url_dict,symbol3,'etf')]['Symbol'].to_list()
        calculate_plot_rolling_correlation_benchmark(etf_ticker=symbol3, stock_ticker_list=list3, market_date_start=market_date_start, market_date_end=market_date_end, market_corr_rolling_window=market_corr_rolling_window)
        
    st.divider()
# Tab 1: Market Condition ####################################################################################

# Tab 2: Reference ####################################################################################
with tab2:
    st.subheader('Reference Stocks')
    st.write(f'Displaying {len(df_tickers):,} companies.')
    st.dataframe(df_tickers[key_cols],use_container_width=True,hide_index=True)
    st.divider()
# Tab 2: Reference ####################################################################################

# Tab 3: Stock Data ####################################################################################
with tab3:
    # Pull data from Alpaca ####################################################################################
    st.subheader('Price Chart')
    
    # Display dataframe
    # Pull data from Alpaca
    if frequency == 'Day': request_params = StockBarsRequest( symbol_or_symbols=selected_stocks_list, timeframe=TimeFrame.Day, start=date_start, end=date_end,adjustment='split')
    elif frequency == 'Hour': request_params = StockBarsRequest( symbol_or_symbols=selected_stocks_list, timeframe=TimeFrame.Hour, start=date_start, end=date_end,adjustment='split')
    elif frequency == 'Minute': request_params = StockBarsRequest( symbol_or_symbols=selected_stocks_list, timeframe=TimeFrame.Minute, start=date_start, end=date_end,adjustment='split')
    # Print close price chart
    if len(selected_stocks_list)==0: st.write('No Stocks Selected')
    else: 
        bars = client.get_stock_bars(request_params)
        data = bars.df.reset_index(level=0)
        # convert to US Eastern Time
        data.index = data.index.tz_convert('US/Eastern')
        # calcualte time columns
        data['date'] = data.index.date
        data['hour'] = data.index.hour
        data['minute'] = data.index.minute
        # remove non-trading hours if frequency is Hour or Minute
        if frequency == 'Hour' or frequency == 'Minute':
            data = data.between_time('09:30','16:00')
        
        ################ Altair chart ################
        # Encoding Types: Q (quantitative), N (nominal), O (ordinal), T (temporal)
        price_chart = alt.Chart(data.reset_index()).mark_line().encode(
            x=alt.X('timestamp:O', timeUnit=timeUnit, title='Date'),
            y=alt.Y('close:Q',  title='Price', scale=alt.Scale(zero=False)),
            color='symbol:N'
        )
        st.altair_chart(price_chart, use_container_width=True)
        ################ Altair chart ################
    st.divider()

    # Display Sample Data ####################################################################################       
    sample_data = st.expander('Show Sample Data')
    sample_data.dataframe(data,use_container_width=True)
# Tab 3: Stock Data ####################################################################################

# Tab 4: Correlation Analysis ####################################################################################
with tab4:
    # Correlation Analysis ####################################################################################
    st.subheader('Correlation Analysis')
    df = data.copy()

    # make 2 columns
    col1, col2, col3 = st.columns([4, 0.1, 6])
    with col1:
        ################ Cluster Heatmap ################
        st.subheader('Cluster Heatmap')
        df_return = df.pivot_table(index='timestamp',columns='symbol',values='close',aggfunc='mean').sort_index(ascending=True).pct_change()
        heatmap = sns.clustermap(df_return.corr(), cmap='RdBu', vmin=-1, vmax=1, center=0, linecolor='white', linewidths=1, annot=True, fmt=".2f", annot_kws={'fontsize':14})
        st.pyplot(heatmap, bbox_inches='tight', dpi=300) # Display the plot in Streamlit
        ################ Cluster Heatmap ################
    with col3:
        ################ Pair Correlation ################
        st.subheader('Pair correlation')
        # select a specific pair
        col1, col2, col3 = st.columns(3)
        with col1:
            selected1 = st.selectbox('Select Stock 1', selected_stocks_list, index=0)
            index1 = get_key_from_value(url_dict, selected1, 'etf')
        with col2: 
            selected2 = st.selectbox('Select Stock 2', [x for x in selected_stocks_list if x != selected1] , index=0)
            index2 = get_key_from_value(url_dict, selected2, 'etf')
        with col3: rolling_window = st.number_input(f'Rolling Window - {frequency}s', min_value=1, max_value=120, value=21, step=1)
        # plot rolling correlation
        df_return['rolling_corr'] = df_return[selected1].rolling(rolling_window).corr(df_return[selected2])
        
        ################ Altair chart ################
        # Encoding Types: Q (quantitative), N (nominal), O (ordinal), T (temporal)
        corr_chart = alt.Chart(df_return.reset_index(),title=f'Rolling {rolling_window} {frequency} correlation - {index1} vs. {index2}').mark_line(interpolate='step-after').encode(
            x=alt.X('timestamp:O', timeUnit=timeUnit, title='Date'),
            y=alt.Y('rolling_corr:Q',  title='Correlation', scale=alt.Scale(domain=[-1, 1], zero=False)),
        )
        lines = alt.Chart(pd.DataFrame({'y': [-1,0,1]})).mark_rule().encode(y='y')
        st.altair_chart(corr_chart+lines, use_container_width=True)
        ################ Altair chart ################
        ################ Pair Correlation ################
# Tab 4: Correlation Analysis ####################################################################################
