import pandas as pd

df = pd.read_csv("fertilizer_cleaned.csv")

print(df['Recommended_Fertilizer'].unique())
