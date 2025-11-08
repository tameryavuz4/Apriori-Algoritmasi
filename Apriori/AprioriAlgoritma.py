import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules




# veriseti
data = {
    'Süt': [1, 1, 0, 1, 0, 1, 0, 1, 1, 0],
    'Ekmek': [1, 0, 1, 1, 1, 0, 1, 0, 1, 1],
    'Yağ': [0, 1, 1, 1, 1, 1, 0, 1, 1, 0],
    'Yumurta': [1, 0, 0, 1, 1, 1, 1, 0, 0, 1],
    'Peynir': [0, 1, 1, 0, 1, 1, 0, 1, 0, 1],
    'Bal': [0, 1, 0, 1, 0, 1, 1, 1, 0, 0],
    'Soda': [1, 0, 1, 0, 1, 0, 1, 0, 1, 1],
    'Meyve Suyu': [1, 1, 1, 1, 1, 0, 1, 0, 1, 1]
}
df = pd.DataFrame(data)

# öğe kümeleri oluşturma
frequent_itemsets = apriori(df, min_support=0.5, use_colnames=True)

# öğe kümeleri oluşturma
frequent_itemsets = apriori(df, min_support=0.5, use_colnames=True)

# İlişkilendirme kuralları
rules = association_rules(frequent_itemsets, metric='confidence', min_threshold=0.6)
print(rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']])

