# Import libraries
import pandas as pd
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules
import matplotlib.pyplot as plt
import seaborn as sns
from mlxtend.preprocessing import TransactionEncoder

# Sample dataset
data = [['milk', 'bread', 'nuts'],
        ['milk', 'bread', 'diapers', 'beer'],
        ['milk', 'nuts', 'diapers'],
        ['bread', 'nuts', 'beer'],
        ['milk', 'bread', 'nuts']]

# Convert data to a DataFrame
te = TransactionEncoder()
te_ary = te.fit(data).transform(data)
df = pd.DataFrame(te_ary, columns=te.columns_)

# One-hot encode the data
df_encoded = df.astype(int)

# Find frequent itemsets
frequent_itemsets = apriori(df_encoded, min_support=0.5, use_colnames=True)

# Generate association rules
rules = association_rules(frequent_itemsets, metric='lift', min_threshold=1.0)

# Visualize frequent itemsets
plt.figure(figsize=(8, 4))
sns.barplot(x='support', y='itemsets', data=frequent_itemsets)
plt.xlabel('Support')
plt.ylabel('Itemsets')
plt.title('Frequent Itemsets')
plt.show()

# Visualize association rules
plt.figure(figsize=(8, 4))
sns.scatterplot(x='support', y='confidence', size='lift', data=rules)
plt.xlabel('Support')
plt.ylabel('Confidence')
plt.title('Association Rules')
plt.show()

# Filter rules by confidence and lift
filtered_rules = rules[(rules['confidence'] >= 0.7) & (rules['lift'] > 1.2)]

# Display the filtered rules
print("Filtered Association Rules:")
print(filtered_rules)
