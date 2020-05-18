import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv(r'Google_Stock_Price_Train.csv')

df.head()

plt.plot(df.Date, df.Open, 'b-')
plt.show()