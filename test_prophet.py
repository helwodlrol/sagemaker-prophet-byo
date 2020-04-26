import pandas as pd
from fbprophet import Prophet

df = pd.read_csv('data/shop.csv')
df['ds'] = pd.to_datetime(df['ds'], unit='s')
m = Prophet()
m.fit(df)
future = m.make_future_dataframe(periods=1440, freq='1min', include_history=False)

future.to_csv('container/local_test/payload.csv', header=True)

t_future = pd.read_csv('payload.csv')
forecast = m.predict(t_future)
print(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail())

# fig1 = m.plot(forecast)
# fig2 = m.plot_components(forecast)
