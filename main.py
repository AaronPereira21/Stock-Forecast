import streamlit as st
from datetime import date
import yfinance as yf
from prophet import Prophet
from prophet.plot import plot_plotly
from plotly import graph_objs as go

START = '2010-01-01'
TODAY = date.today().strftime("%Y-%m-%d")

st.title("Stock Predictor")

@st.cache_data
def search_stocks(query):
    """
    Searches for stocks based on a query.
    Returns a list of matching tickers and their names.
    """
    ticker_data = yf.Ticker(query)
    try:
        # Fetch stock information
        info = ticker_data.info
        return {"symbol": info["symbol"], "name": info.get("longName", query)}
    except Exception as e:
        return None


# Implement a search box for stocks
query = st.text_input("Search for a stock ticker (e.g., AAPL, TSLA, etc.)", "AAPL")
if query:
    stock_info = search_stocks(query)
    if stock_info:
        st.write(f"Selected Stock: {stock_info['symbol']} - {stock_info['name']}")
        selected_stock = stock_info['symbol']
    else:
        st.error("Invalid stock ticker or no data available.")
        st.stop()
else:
    st.warning("Enter a stock ticker to search.")

n_years = st.slider("Years of Prediction", 1, 4)
period = n_years * 365

@st.cache_data
def load_data(ticker):
    data = yf.download(ticker, START, TODAY)
    data.reset_index(inplace=True)
    return data

data_load_state = st.text("Loading data...")
data = load_data(selected_stock)
data_load_state.text("Loading data...done!")

st.subheader('Raw data')
st.write(data.tail())

def plot_raw_data():
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Open'], name='stock_open'))
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name='stock_close'))
    fig.layout.update(title_text="Time Series Data", xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)

plot_raw_data()

# Prepare data for forecasting
df_train = data[['Date', 'Close']]
df_train = df_train.rename(columns={"Date": "ds", "Close": "y"})

m = Prophet()
m.fit(df_train)
future = m.make_future_dataframe(periods=period)
forecast = m.predict(future)

st.subheader('Forecast Data')
st.write(forecast.tail())

st.write('Forecast Data')
fig1 = plot_plotly(m, forecast)
st.plotly_chart(fig1)

st.write('Forecast Components')
fig2 = m.plot_components(forecast)
st.write(fig2)
