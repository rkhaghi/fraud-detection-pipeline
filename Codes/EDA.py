#%%
import pandas as pd
import matplotlib.pyplot as plt 

df = pd.read_csv('synthetic_transactions.csv')


# %%
#simple edas

df.head()
df.info()
df.describe()
# %%
df['is_fraud'].value_counts(normalize=True)*100
# %%
#Plot fraud vs. non-fraud distributions: amount, time-of-day, merchant category, country

df.groupby('is_fraud')['amount'].hist(alpha=0.5)
plt.legend(['Non-Fraud', 'Fraud'])
# %%
#  timestamp
df['timestamp'] = pd.to_datetime(df['timestamp'])
df['hour'] = df['timestamp'].dt.hour
df.groupby('is_fraud')['hour'].hist(alpha=0.5)
# %%
df['merchant_country'].value_counts(normalize=True).head(10).plot(kind='bar')
# %%
df['merchant_category'].value_counts(normalize=True).head(10).plot(kind='bar')
# %%
#order the merchant category by fraud rate
fraud_rate_by_category = df.groupby('merchant_category')['is_fraud'].mean()
# %%
#order by maximum value rate
fraud_rate_by_category = fraud_rate_by_category.sort_values(ascending=False)
# %%
##Understand each of the 5 fraud patterns in the data by mcc and merchant category

# Fraud RATE per category (what % of transactions are fraud)
df.groupby('merchant_category')['is_fraud'].mean().sort_values(ascending=False)

# Fraud COUNT per category (volume of fraud cases)
df.groupby('merchant_category')['is_fraud'].sum().sort_values(ascending=False)

# Both together
df.groupby('merchant_category').agg(
    fraud_rate=('is_fraud', 'mean'),
    fraud_count=('is_fraud', 'sum'),
    total_txns=('is_fraud', 'count'),
    average_amount=('amount', 'mean'),
    max_amount=('amount', 'max'),
    min_amount=('amount', 'min')
).sort_values('fraud_rate', ascending=False)


# %%
df.groupby('merchant_category')['amount'].mean().sort_values(ascending=False)
df.groupby('merchant_category')['amount'].max().sort_values(ascending=False)
# %%
