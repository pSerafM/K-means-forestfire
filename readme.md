# K-Means com Hadoop e dados de incêndios

O programa usa **PySpark MLlib** para padronizar os dados e treinar os modelos
K-Means distribuídos. O CSV é lido por uma URI do HDFS; caminhos locais são
recusados. O cálculo Silhouette usa a distância Euclidiana padrão do
scikit-learn sobre os vetores e rótulos produzidos pelo Spark. As correlações
de Pearson são calculadas no Spark usando os valores numéricos originais.

## Pré-requisitos

- Hadoop/HDFS acessível e configurado no ambiente que executará o programa.
- Apache Spark com suporte a PySpark e integração com o Hadoop/YARN. A versão
  do pacote PySpark deve corresponder à versão do Spark do cluster.
- Python com as dependências listadas em `requirements.txt`.

Instale as dependências Python no ambiente do driver:

```powershell
python -m pip install -r requirements.txt
```

## Enviar o CSV ao HDFS e executar

Execute os comandos abaixo em um terminal com `hdfs` e `spark-submit`
configurados para o cluster. Troque `<usuario>` pelo seu usuário HDFS:

```powershell
hdfs dfs -mkdir -p /user/<usuario>/forestfire
hdfs dfs -put -f .\forestfireinput.csv /user/<usuario>/forestfire/forestfireinput.csv
spark-submit --master yarn --deploy-mode client src\main.py --input hdfs:///user/<usuario>/forestfire/forestfireinput.csv --output .\resultados_kmeans
```

O script testa `K` de 2 a 10, incluindo as 11 colunas numéricas (`X`, `Y`,
`FFMC`, `DMC`, `DC`, `ISI`, `temp`, `RH`, `wind`, `rain` e `area`). As colunas
categóricas `month` e `day` não entram no K-Means nem nas correlações. O
Silhouette é calculado com distância Euclidiana; como o conjunto atual tem
517 linhas, os vetores padronizados e as atribuições são enviados ao driver
para esse cálculo. O treinamento K-Means, a padronização e as correlações são
executados no Spark.

## Arquivos gerados

A pasta local indicada por `--output` conterá:

- `silhouette_por_k.csv` e `silhouette_por_k.png`: pontuação para cada K.
- `correlacoes_por_k_e_cluster.csv`: matriz completa (incluindo valores
  simétricos) para cada cluster de cada K avaliado.
- `correlacoes_por_cluster_melhor_k.png`: mapas de calor das correlações no
  melhor K.
- `resumo_clusters_melhor_k.csv`: tamanhos, médias das variáveis, mediana da
  área e proporção de registros com área queimada positiva.
- `atribuicoes_melhor_k.csv`: valores numéricos e cluster atribuído por linha.
- `perfil_dos_clusters_melhor_k.png`: médias por cluster comparadas em
  desvios-padrão em relação à média geral.
- `area_por_cluster_melhor_k.png`: distribuição de `log(1 + area)` por
  cluster, para facilitar a leitura dos valores extremos.

O melhor K é o que obtiver a maior pontuação Silhouette. Use também os tamanhos
dos clusters e as matrizes de correlação para contextualizar a conclusão; a
maior pontuação não prova, por si só, que os grupos sejam úteis ou expliquem
causalmente os incêndios.
