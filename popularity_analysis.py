!pip install pandas statsmodels scipy
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.anova import anova_lm

# Загрузка данных
data = pd.read_csv('song_data.csv')


print(data.head())
print(data.columns)
y = data['song_popularity']
X = data[['song_duration_ms', 'danceability', 'energy']]
X = sm.add_constant(X)

# Построение модели линейной регрессии
model = sm.OLS(y, X).fit()

print(model.summary())

# Коэффициент детерминации
r_squared = model.rsquared
print(f"Коэффициент детерминации R^2: {r_squared}")

# Проверка гипотезы 1: Чем больше "энергичность", тем больше популярность
energy_coef = model.params['energy']
energy_pvalue = model.pvalues['energy']
print(f"Коэффициент при energy: {energy_coef}")
print(f"P-value для energy: {energy_pvalue}")
if energy_pvalue < 0.05:
    print("Гипотеза о влиянии энергичности на популярность подтверждается.")
else:
    print("Гипотеза о влиянии энергичности на популярность не подтверждается.")

# Проверка гипотезы 2: Популярность зависит от "танцевальности"
danceability_coef = model.params['danceability']
danceability_pvalue = model.pvalues['danceability']
print(f"Коэффициент при danceability: {danceability_coef}")
print(f"P-value для danceability: {danceability_pvalue}")
if danceability_pvalue < 0.05:
    print("Гипотеза о влиянии танцевальности на популярность подтверждается.")
else:
    print("Гипотеза о влиянии танцевальности на популярность не подтверждается.")

# Проверка гипотезы 3: Популярность зависит одновременно от продолжительности и танцевальности

X_full = data[['song_duration_ms', 'danceability', 'energy']]
X_full = sm.add_constant(X_full)
model_full = sm.OLS(y, X_full).fit()

X_restricted = data[['energy']]
X_restricted = sm.add_constant(X_restricted)
model_restricted = sm.OLS(y, X_restricted).fit()

anova_results = anova_lm(model_restricted, model_full)

f_stat = anova_results['F'][1]
f_pvalue = anova_results['Pr(>F)'][1]
print(f"F-статистика: {f_stat}")
print(f"P-value для F-теста: {f_pvalue}")
if f_pvalue < 0.05:
    print("Гипотеза о влиянии продолжительности и танцевальности на популярность подтверждается.")
else:
    print("Гипотеза о влиянии продолжительности и танцевальности на популярность не подтверждается.")

