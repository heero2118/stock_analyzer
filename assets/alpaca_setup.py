# Alpaca API ####################################################################################
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame

# Alpaca keys
api_key = 'AKARY3S28GJ1O05GWRVX'
secret_key = 'WhECbTRozVhvjCfxjEzkMgbtrlD0NTCwnFgfftFF'
client = StockHistoricalDataClient(api_key=api_key, secret_key=secret_key)