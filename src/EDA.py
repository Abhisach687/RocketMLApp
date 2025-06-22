# %% [markdown]
# Retailrocket Dataset: Essential EDA for Forecasting & Recommendation
# Run in VSCode by cells; no display() calls; clear, non-overlapping labels

# %%
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

plt.rcParams.update({'figure.autolayout': True})

# Load cleaned data
data_dir = Path('../data/interim')
events = pd.read_parquet(data_dir / 'events_clean.parquet')
props  = pd.read_parquet(data_dir / 'item_properties.parquet')
cats   = pd.read_parquet(data_dir / 'category_tree.parquet')

# %% [markdown]
# ## 1. Data Overview & Missingness

# %%
print(f"Events: {events.shape[0]} rows, {events.shape[1]} cols")
print(f"Properties: {props.shape[0]} rows, {props.shape[1]} cols")
print(f"Categories: {cats.shape[0]} rows, {cats.shape[1]} cols")

for name, df in [('Events', events), ('Properties', props), ('Categories', cats)]:
    miss = df.isnull().mean() * 100
    print(f"\n{name} missing (%):")
    print(miss[miss > 0].round(1))

# %% [markdown]
# ## 2. Event-Type Distribution & Time Patterns

# %%
# 2.1 Event types
etype = events['event'].value_counts()
fig, ax = plt.subplots(figsize=(6,4))
etype.plot.bar(ax=ax)
ax.set_title('Event Type Counts')
ax.set_xlabel('Event')
ax.set_ylabel('Count')
ax.tick_params(axis='x', rotation=0)
plt.show()

# 2.2 Daily trend
fig, ax = plt.subplots(figsize=(8,3))
daily = events.groupby(events['timestamp'].dt.date).size()
daily.plot(ax=ax)
ax.set_title('Daily Event Volume')
ax.set_xlabel('Date')
ax.set_ylabel('Count')
fig.autofmt_xdate()
plt.show()

# 2.3 Hourly pattern
fig, ax = plt.subplots(figsize=(6,3))
hourly = events['timestamp'].dt.hour.value_counts().sort_index()
hourly.plot.bar(ax=ax)
ax.set_title('Hourly Event Volume')
ax.set_xlabel('Hour')
ax.set_ylabel('Count')
plt.show()

# %% [markdown]
# ## 3. Sessionization & Funnel Metrics

# %%
# Define sessions (30-min inactivity)
ev = events.sort_values(['visitorid','timestamp']).copy()
ev['prev'] = ev.groupby('visitorid')['timestamp'].shift()
ev['delta'] = (ev['timestamp'] - ev['prev']).dt.total_seconds() / 60
ev['new_session'] = ev['delta'].gt(30) | ev['delta'].isna()
ev['session'] = ev.groupby('visitorid')['new_session'].cumsum()

# Session length distribution
sess_len = ev.groupby('session').size()
fig, ax = plt.subplots(figsize=(6,4))
sess_len[sess_len <= sess_len.quantile(0.95)].plot.hist(bins=30, ax=ax)
ax.set_title('Session Length (<=95th percentile)')
ax.set_xlabel('Events per Session')
ax.set_ylabel('Frequency')
plt.show()

# Funnel conversion
funnel = ev.pivot_table(index='session', columns='event', values='visitorid', aggfunc='size', fill_value=0)
r1 = ((funnel['addtocart']>0) & (funnel['view']>0)).mean()
r2 = ((funnel['transaction']>0) & (funnel['addtocart']>0)).mean()
print(f"View→AddToCart: {r1:.1%}")
print(f"AddToCart→Transaction: {r2:.1%}")

# %% [markdown]
# ## 4. Item Popularity & Long-Tail

# %%
item_counts = events['itemid'].value_counts()
top_n = 15
fig, ax = plt.subplots(figsize=(6,5))
item_counts.head(top_n).sort_values().plot.barh(ax=ax)
ax.set_title(f'Top {top_n} Items by Count')
ax.set_xlabel('Count')
plt.show()

fig, ax = plt.subplots(figsize=(6,4))
cum = item_counts.cumsum() / item_counts.sum()
cum.plot(ax=ax)
ax.axhline(0.8, color='red', linestyle='--')
ax.set_title('Cumulative Coverage by Item Rank')
ax.set_xlabel('Rank')
ax.set_ylabel('Cumulative %')
plt.show()

# %% [markdown]
# ## 5. Category Insights (with Names)

# %%
# Map item→categoryid→name
cat_map = props[props['property']=='categoryid'][['itemid','value']]
cat_map.columns=['itemid','categoryid']
ec = events.merge(cat_map, on='itemid', how='left')
ec['cat_name'] = ec['categoryid'].fillna(-1).astype(int).map(lambda x: f"Category {x}" if x>=0 else 'Unknown')

fig, ax = plt.subplots(figsize=(6,5))
ec['cat_name'].value_counts().head(top_n).sort_values().plot.barh(ax=ax)
ax.set_title('Top Categories by Count')
ax.set_xlabel('Count')
plt.show()
