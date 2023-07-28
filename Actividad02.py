"""
Seminario de Inteligencia Artificial 2
Actividad 02
"""

import pandas as pd

df = pd.read_csv('countries.csv')
df = df.rename(columns={'pop':'population'})
#pd.set_option("display.max_rows", None, "display.max_columns", None)

print("Pregunta 1")
r = df[df.country == 'Tunisia'].continent.iat[0]
print(r)

print("Pregunta 2")
r = list(df[df.lifeExp > 80][df.year == 2007].country)
print(r)

print("Pregunta 3")
r = df[df.gdpPercap == max(df[df.continent == 'Americas'].gdpPercap)][df.continent == 'Americas'].country.iat[0]
print(r)

print("Pregunta 4")
r = df.country.iat[df[(df.country == 'Venezuela') | (df.country == 'Paraguay')][df.year == 1967].population.idxmax()]
print(r)

print("Pregunta 5")
r = df[df.country == 'Panama'][df.lifeExp > 60].year.iat[0]
print(r)

print("Pregunta 6")
r = df[df.continent == 'Africa'][df.year == 2007].lifeExp.mean()
print(r)

print("Pregunta 7")
r = df[df.year == 2002].reset_index()
r2 = df[df.year == 2007].reset_index()
r3 = list(r[r.gdpPercap > r2.gdpPercap].country)
print(r3)

print("Pregunta 8")
r = df[df.year == 2007][df.population == max(df.population)].country.iat[0]
print(r)

print("Pregunta 9")
r = df[df.continent == 'Americas'][df.year == 2007].sum().population
print(r)

print("Pregunta 10")
r = df[df.year == 2007].groupby(['continent']).sum().population.sort_index().keys()[-1]
print(r)

print("Pregunta 11")
r = df[df.continent == 'Europe'].mean().gdpPercap
print(r)

print("Pregunta 12")
r = df[df.continent == 'Europe'][df.population > 70000000].sort_values(by=['year']).country.iat[0]
print(r)

