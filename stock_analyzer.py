import streamlit as st

# Set Page Name and Emoji ####################################################################################
st.set_page_config(page_title='Stock Analyzer',layout='wide',page_icon='📈')
tab1, tab2, tab3 = st.tabs(['🕵️‍♂️ Screener','📋 Data','📊 Analsysis'])

with tab1:
    # find tickers
    import financedatabase as fd

    # Initialize the Equities database
    equities = fd.Equities()
    # Obtain all countries from the database
    countries = ['United States', 'Afghanistan', 'Anguilla', 'Argentina', 'Australia', 'Austria',
        'Azerbaijan', 'Bahamas', 'Bangladesh', 'Barbados', 'Belgium', 'Belize', 'Bermuda', 'Botswana', 'Brazil',
        'British Virgin Islands', 'Cambodia', 'Canada', 'Cayman Islands', 'Chile', 'China', 'Colombia', 'Costa Rica', 'Cyprus',
        'Czech Republic', 'Denmark', 'Dominican Republic', 'Egypt', 'Estonia', 'Falkland Islands', 'Finland', 'France',
        'French Guiana', 'Gabon', 'Georgia', 'Germany', 'Ghana', 'Gibraltar', 'Greece', 'Greenland', 'Guernsey', 'Hong Kong',
        'Hungary', 'Iceland', 'India', 'Indonesia', 'Ireland', 'Isle of Man', 'Israel', 'Italy', 'Ivory Coast', 'Japan', 'Jersey',
        'Jordan', 'Kazakhstan', 'Kenya', 'Kyrgyzstan', 'Latvia', 'Liechtenstein', 'Lithuania', 'Luxembourg', 'Macau', 'Macedonia',
        'Malaysia', 'Malta', 'Mauritius', 'Mexico', 'Monaco', 'Mongolia', 'Montenegro', 'Morocco', 'Mozambique', 'Myanmar', 'Namibia',
        'Netherlands', 'Netherlands Antilles', 'New Zealand', 'Nigeria', 'Norway', 'Panama', 'Papua New Guinea', 'Peru', 'Philippines',
        'Poland', 'Portugal', 'Qatar', 'Reunion', 'Romania', 'Russia', 'Saudi Arabia', 'Senegal', 'Singapore', 'Slovakia', 'Slovenia',
        'South Africa', 'South Korea', 'Spain', 'Suriname', 'Sweden', 'Switzerland', 'Taiwan', 'Tanzania', 'Thailand', 'Turkey',
        'Ukraine', 'United Arab Emirates', 'United Kingdom', 'Uruguay', 'Vietnam', 'Zambia']
    # Obtain all sectors from the database
    sectors = ['Communication Services', 'Consumer Discretionary', 'Consumer Staples', 'Energy', 'Financials',
            'Health Care', 'Industrials', 'Information Technology', 'Materials', 'Real Estate', 'Utilities']
    # Obtain all industry groups from the database
    industry_groups = ['Automobiles & Components', 'Banks', 'Capital Goods', 'Commercial & Professional Services',
        'Consumer Durables & Apparel', 'Consumer Services', 'Diversified Financials', 'Energy', 'Food & Staples Retailing',
        'Food, Beverage & Tobacco', 'Health Care Equipment & Services', 'Household & Personal Products', 'Insurance', 'Materials',
        'Media & Entertainment', 'Pharmaceuticals, Biotechnology & Life Sciences', 'Real Estate',
        'Retailing', 'Semiconductors & Semiconductor Equipment', 'Software & Services', 'Technology Hardware & Equipment',
        'Telecommunication Services', 'Transportation', 'Utilities']
    # Obtain all industry from the database
    industries = ['Aerospace & Defense', 'Air Freight & Logistics', 'Airlines', 'Auto Components', 'Automobiles', 'Banks', 'Beverages',
        'Biotechnology', 'Building Products', 'Capital Markets', 'Chemicals', 'Commercial Services & Supplies',
        'Communications Equipment', 'Construction & Engineering', 'Construction Materials', 'Consumer Finance', 'Distributors',
        'Diversified Consumer Services', 'Diversified Financial Services', 'Diversified Telecommunication Services', 'Electric Utilities',
        'Electrical Equipment', 'Electronic Equipment, Instruments & Components', 'Energy Equipment & Services', 'Entertainment',
        'Equity Real Estate Investment Trusts (REITs)', 'Food & Staples Retailing', 'Food Products', 'Gas Utilities',
        'Health Care Equipment & Supplies', 'Health Care Providers & Services', 'Health Care Technology',
        'Hotels, Restaurants & Leisure', 'Household Durables', 'Household Products', 'IT Services',
        'Independent Power and Renewable Electricity Producers', 'Industrial Conglomerates', 'Insurance',
        'Interactive Media & Services', 'Internet & Direct Marketing Retail', 'Machinery', 'Marine',
        'Media', 'Metals & Mining', 'Multi-Utilities', 'Oil, Gas & Consumable Fuels', 'Paper & Forest Products',
        'Pharmaceuticals', 'Professional Services', 'Real Estate Management & Development', 'Road & Rail', 'Semiconductors & Semiconductor Equipment', 'Software',
        'Specialty Retail', 'Technology Hardware, Storage & Peripherals', 'Textiles, Apparel & Luxury Goods', 'Thrifts & Mortgage Finance',
        'Tobacco', 'Trading Companies & Distributors', 'Transportation Infrastructure', 'Water Utilities']
    # Create list of market cap sizes
    market_cap_sizes = ['Large Cap', 'Mid Cap', 'Small Cap', 'Micro Cap', 'Nano Cap']

    # Selection boxes ####################################################################################
    with st.container():
        st.title("Stock Screener using financedatabase")
        col1,col2,col3,col4,col5=st.columns(5)
        with col1:
            selected_country = st.selectbox('Select Country',countries,index=0)
        with col2:
            selected_sector = st.selectbox('Select Sector',sectors,index=None)
            selected_sector = selected_sector if selected_sector is not None else ""
        with col3:
            selected_industry_group = st.selectbox('Select Industry Group',industry_groups,index=None)
            selected_industry_group = selected_industry_group if selected_industry_group is not None else ""
        with col4:
            selected_industry = st.selectbox('Select Industry',industries,index=None)
            selected_industry = selected_industry if selected_industry is not None else ""
        with col5:
            selected_market_cap = st.multiselect('Select Market Cap Size',market_cap_sizes,default=['Large Cap','Mid Cap'])

        # Display filtered stocks ####################################################################################
        key_cols = ['symbol','name','summary','currency','exchange','market','market_cap','sector','industry_group','industry','country','state','city','zipcode','website']
        selected_stocks = equities.select(country=selected_country, sector=selected_sector, industry_group=selected_industry_group, industry=selected_industry)
        selected_stocks = selected_stocks[selected_stocks['market_cap'].isin(selected_market_cap)].reset_index()
        st.write(f'Displaying {len(selected_stocks)} stocks.')
        st.dataframe(selected_stocks[key_cols])
        st.divider()

    # key word search ####################################################################################
    with st.container():
        st.title("Filter further with summary keywords")
        keyword = st.text_input('Enter keyword',placeholder='')
        selected_stocks_w_keyword = selected_stocks[selected_stocks['summary'].str.contains(keyword,case=False)==True]
        st.write(f'Displaying {len(selected_stocks_w_keyword)} stocks.')
        st.dataframe(selected_stocks_w_keyword[key_cols])
        st.divider()

with tab2:
    import pandas as pd
    from datetime import datetime

    # Alpaca keys
    api_key = 'AKARY3S28GJ1O05GWRVX'
    secret_key = 'WhECbTRozVhvjCfxjEzkMgbtrlD0NTCwnFgfftFF'

    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame

    client = StockHistoricalDataClient(api_key=api_key, secret_key=secret_key)

    request_params = StockBarsRequest(
                            symbol_or_symbols=['PLTR','AAPL'],
                            timeframe=TimeFrame.Day,
                            start=datetime(2025, 1, 1),
                            end=datetime(2025, 3, 23)
                    )

    bars = client.get_stock_bars(request_params)