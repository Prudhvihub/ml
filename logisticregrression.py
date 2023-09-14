import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Load the data
df = pd.read_csv("https://stats.idre.ucla.edu/stat/data/binary.csv")

# Convert rank to a categorical variable
df['rank'] = df['rank'].astype('category')

# Fit the logistic regression model
logit = smf.glm(formula = 'admit ~ gre + gpa + rank', data = df, family = sm.families.Binomial()).fit()

# Print the summary
print(logit.summary())

# Predict the probability of admission for a student with GRE=790, GPA=3.8, and rank=1
x = pd.DataFrame({'gre': [790], 'gpa': [3.8], 'rank': ['1']})
p = logit.predict(x)
print(p)
