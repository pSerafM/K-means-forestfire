import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

colunas = [
    "X", "Y", "month", "day", "FFMC", "DMC", "DC",
    "ISI", "temp", "RH", "wind", "rain", "area"
]
df = pd.read_csv("forestfireinput.csv", header=None, names=colunas)

# Comece só com variáveis numéricas; deixe mês, dia e área de fora.
variaveis = ["X", "Y", "FFMC", "DMC", "DC", "ISI", "temp", "RH", "wind", "rain"]
X = StandardScaler().fit_transform(df[variaveis])

# Compare alguns números de grupos (k).
pontuacoes = {}
for k in range(2, 9):
    modelo = KMeans(n_clusters=k, random_state=42, n_init=10)
    grupos = modelo.fit_predict(X)
    pontuacoes[k] = silhouette_score(X, grupos)

melhor_k = max(pontuacoes, key=pontuacoes.get)
modelo = KMeans(n_clusters=melhor_k, random_state=42, n_init=10)
df["cluster"] = modelo.fit_predict(X)

print("Silhouette por k:", pontuacoes)
print(df.groupby("cluster")[variaveis].mean().round(2))