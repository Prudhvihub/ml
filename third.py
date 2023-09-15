import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

df = pd.read_csv("https://stats.idre.ucla.edu/stat/data/binary.csv")
df

df.info()

df.isna().sum()

df.describe()
unique_ranks = df['rank'].unique()
print(unique_ranks)
unique_ranks = pd.crosstab(df['admit'], df['rank'])
print(unique_ranks)
df['rank'] = df['rank'].astype('category')
try:
    logit = smf.glm(formula='admit ~ gre + gpa + rank', data=df, family=sm.families.Binomial()).fit()
    # Print the summary
    print(logit.summary())
except Exception as e:
    print(f"Error: {e}")
    
df['rank'] = pd.Categorical(df['rank'], categories=[1, 2, 3, 4])
logit = smf.glm(formula='admit ~ gre + gpa + rank', data=df, family=sm.families.Binomial()).fit()

x = pd.DataFrame({'gre': [790], 'gpa': [3.8], 'rank': [1]})  
p = logit.predict(x)
print(f"Probability of admission : {p[0]:.4f}")