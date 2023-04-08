import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

data = pd.read_csv("../../COMP4948/assignment2/telecom_churn.csv")
print(data)
print(data.describe().T)

data['Churn'].value_counts().plot(kind='bar', xlabel="Leaves company?", title="Customer Churn")
data.plot.bar("Churn", "AccountWeeks")
plt.show()

sns.catplot(x = "x",       # x variable name
            y = "y",       # y variable name
            hue = "type",  # group variable name
            data = df,     # dataframe to plot
            kind = "bar")
