import argparse
import math
from itertools import combinations_with_replacement
from pathlib import Path
from urllib.parse import urlparse

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
from pyspark import StorageLevel
from pyspark.ml.clustering import KMeans
from pyspark.ml.feature import StandardScaler, VectorAssembler
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType, StringType, StructField, StructType
from sklearn.metrics import silhouette_score

K_VALUES = range(2, 11)
NUMERIC_COLUMNS = [
    "X",
    "Y",
    "FFMC",
    "DMC",
    "DC",
    "ISI",
    "temp",
    "RH",
    "wind",
    "rain",
    "area",
]
CSV_SCHEMA = StructType(
    [
        StructField("X", DoubleType(), True),
        StructField("Y", DoubleType(), True),
        StructField("month", StringType(), True),
        StructField("day", StringType(), True),
        StructField("FFMC", DoubleType(), True),
        StructField("DMC", DoubleType(), True),
        StructField("DC", DoubleType(), True),
        StructField("ISI", DoubleType(), True),
        StructField("temp", DoubleType(), True),
        StructField("RH", DoubleType(), True),
        StructField("wind", DoubleType(), True),
        StructField("rain", DoubleType(), True),
        StructField("area", DoubleType(), True),
    ]
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Executa K-Means distribuído no Spark sobre um CSV no HDFS."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="URI HDFS do CSV, por exemplo hdfs:///user/aluno/forestfireinput.csv",
    )
    parser.add_argument(
        "--output",
        default="resultados_kmeans",
        help="Pasta local para tabelas e gráficos (padrão: resultados_kmeans).",
    )
    return parser.parse_args()


def validate_hdfs_uri(input_uri: str) -> None:
    scheme = urlparse(input_uri).scheme.lower()
    if scheme not in {"hdfs", "viewfs"}:
        raise ValueError(
            "A entrada deve estar no Hadoop: informe uma URI hdfs:// ou viewfs://."
        )


def read_numeric_data(spark: SparkSession, input_uri: str) -> DataFrame:
    data = (
        spark.read.option("header", "false")
        .option("mode", "FAILFAST")
        .schema(CSV_SCHEMA)
        .csv(input_uri)
        .select(*NUMERIC_COLUMNS)
    )
    row_count = data.count()
    if row_count < max(K_VALUES) + 1:
        raise ValueError(
            f"Foram lidas {row_count} linhas; são necessárias pelo menos "
            f"{max(K_VALUES) + 1} para avaliar K de 2 a 10."
        )

    missing_condition = F.lit(False)
    for column in NUMERIC_COLUMNS:
        missing_condition = missing_condition | F.col(column).isNull()
    missing_count = data.filter(missing_condition).count()
    if missing_count:
        raise ValueError(
            f"O CSV contém {missing_count} linhas com valores numéricos ausentes."
        )
    return data


def correlation_rows(data: DataFrame, k: int) -> list[dict[str, object]]:
    pairs = list(combinations_with_replacement(NUMERIC_COLUMNS, 2))
    aggregations = [
        F.corr(F.col(first), F.col(second)).alias(f"corr_{index}")
        for index, (first, second) in enumerate(pairs)
    ]
    aggregations.insert(0, F.count("*").alias("cluster_size"))

    rows = []
    for cluster_row in data.groupBy("cluster").agg(*aggregations).collect():
        cluster = int(cluster_row["cluster"])
        for index, (first, second) in enumerate(pairs):
            value = cluster_row[f"corr_{index}"]
            rows.append(
                {
                    "k": k,
                    "cluster": cluster,
                    "cluster_size": cluster_row["cluster_size"],
                    "variavel_1": first,
                    "variavel_2": second,
                    "correlacao": value,
                }
            )
            if first != second:
                rows.append(
                    {
                        "k": k,
                        "cluster": cluster,
                        "cluster_size": cluster_row["cluster_size"],
                        "variavel_1": second,
                        "variavel_2": first,
                        "correlacao": value,
                    }
                )
    return rows


def plot_silhouette(scores: pd.DataFrame, output_dir: Path) -> None:
    figure, axis = plt.subplots(figsize=(9, 5))
    bars = axis.bar(scores["k"], scores["silhouette"], color="#3478a8")
    axis.set(
        title="Silhouette por número de clusters",
        xlabel="Número de clusters (K)",
        ylabel="Silhouette (distância Euclidiana)",
        xticks=scores["k"],
    )
    axis.set_ylim(min(0, scores["silhouette"].min() - 0.05), 1)
    axis.grid(axis="y", alpha=0.25)
    axis.bar_label(bars, fmt="%.3f", padding=3)
    figure.tight_layout()
    figure.savefig(output_dir / "silhouette_por_k.png", dpi=180)
    plt.close(figure)


def plot_correlation_matrices(
    correlations: pd.DataFrame, best_k: int, output_dir: Path
) -> None:
    best = correlations[correlations["k"] == best_k]
    clusters = sorted(best["cluster"].unique())
    columns = min(3, len(clusters))
    rows = math.ceil(len(clusters) / columns)
    figure, axes = plt.subplots(
        rows,
        columns,
        figsize=(6 * columns, 5.2 * rows),
        squeeze=False,
    )
    image = None

    for axis, cluster in zip(axes.flat, clusters):
        matrix = (
            best[best["cluster"] == cluster]
            .pivot(index="variavel_1", columns="variavel_2", values="correlacao")
            .reindex(index=NUMERIC_COLUMNS, columns=NUMERIC_COLUMNS)
        )
        image = axis.imshow(matrix, cmap="coolwarm", vmin=-1, vmax=1)
        axis.set(
            title=f"Cluster {cluster}",
            xticks=range(len(NUMERIC_COLUMNS)),
            yticks=range(len(NUMERIC_COLUMNS)),
            xticklabels=NUMERIC_COLUMNS,
            yticklabels=NUMERIC_COLUMNS,
        )
        axis.tick_params(axis="x", labelrotation=90, labelsize=8)
        axis.tick_params(axis="y", labelsize=8)

    for axis in axes.flat[len(clusters) :]:
        axis.set_visible(False)
    figure.colorbar(image, ax=axes.ravel().tolist(), label="Correlação de Pearson")
    figure.suptitle(f"Matrizes de correlação por cluster (K = {best_k})")
    figure.savefig(
        output_dir / "correlacoes_por_cluster_melhor_k.png",
        dpi=180,
        bbox_inches="tight",
    )
    plt.close(figure)


def plot_cluster_profiles(
    assignments: pd.DataFrame, best_k: int, output_dir: Path
) -> None:
    means = assignments.groupby("cluster")[NUMERIC_COLUMNS].mean()
    overall_std = assignments[NUMERIC_COLUMNS].std(ddof=0).replace(0, 1)
    profiles = (means - assignments[NUMERIC_COLUMNS].mean()) / overall_std

    figure, axis = plt.subplots(
        figsize=(max(10, len(NUMERIC_COLUMNS) * 0.9), 5)
    )
    image = axis.imshow(profiles, cmap="coolwarm", aspect="auto", vmin=-2, vmax=2)
    axis.set(
        title=f"Perfil médio padronizado dos clusters (K = {best_k})",
        xlabel="Variável numérica",
        ylabel="Cluster",
        xticks=range(len(NUMERIC_COLUMNS)),
        xticklabels=NUMERIC_COLUMNS,
        yticks=range(len(profiles.index)),
        yticklabels=profiles.index,
    )
    axis.tick_params(axis="x", labelrotation=45)
    figure.colorbar(image, ax=axis, label="Desvios-padrão em relação à média geral")
    figure.tight_layout()
    figure.savefig(output_dir / "perfil_dos_clusters_melhor_k.png", dpi=180)
    plt.close(figure)


def plot_area_distribution(
    assignments: pd.DataFrame, best_k: int, output_dir: Path
) -> None:
    clusters = sorted(assignments["cluster"].unique())
    area_values = [
        assignments.loc[assignments["cluster"] == cluster, "area"].map(math.log1p)
        for cluster in clusters
    ]
    figure, axis = plt.subplots(figsize=(8, 5))
    axis.boxplot(area_values, tick_labels=[str(cluster) for cluster in clusters])
    axis.set(
        title=f"Área queimada por cluster (K = {best_k})",
        xlabel="Cluster",
        ylabel="log(1 + area)",
    )
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(output_dir / "area_por_cluster_melhor_k.png", dpi=180)
    plt.close(figure)


def write_outputs(
    scores: pd.DataFrame,
    correlations: pd.DataFrame,
    assignments: pd.DataFrame,
    output_dir: Path,
) -> int:
    best_k = int(scores.loc[scores["silhouette"].idxmax(), "k"])
    best_score = float(scores.loc[scores["k"] == best_k, "silhouette"].iloc[0])

    summary = assignments.groupby("cluster").agg(
        registros=("area", "size"),
        area_mediana=("area", "median"),
        proporcao_com_area=("area", lambda values: (values > 0).mean()),
    )
    means = assignments.groupby("cluster")[NUMERIC_COLUMNS].mean().add_prefix("media_")
    summary.join(means).reset_index().to_csv(
        output_dir / "resumo_clusters_melhor_k.csv", index=False
    )
    assignments.to_csv(output_dir / "atribuicoes_melhor_k.csv", index=False)
    scores.to_csv(output_dir / "silhouette_por_k.csv", index=False)
    correlations.to_csv(output_dir / "correlacoes_por_k_e_cluster.csv", index=False)

    plot_silhouette(scores, output_dir)
    plot_correlation_matrices(correlations, best_k, output_dir)
    plot_cluster_profiles(assignments, best_k, output_dir)
    plot_area_distribution(assignments, best_k, output_dir)

    print(f"Melhor K: {best_k} (Silhouette Euclidiana: {best_score:.4f})")
    print(f"Resultados salvos em: {output_dir.resolve()}")
    return best_k


def run_analysis(input_uri: str, output_dir: Path) -> None:
    validate_hdfs_uri(input_uri)
    output_dir.mkdir(parents=True, exist_ok=True)

    spark = (
        SparkSession.builder.appName("KMeansForestFire")
        .config("spark.sql.execution.arrow.pyspark.enabled", "false")
        .getOrCreate()
    )
    scaled_data = None
    try:
        numeric_data = read_numeric_data(spark, input_uri)
        assembled_data = VectorAssembler(
            inputCols=NUMERIC_COLUMNS,
            outputCol="assembled_features",
            handleInvalid="error",
        ).transform(numeric_data)
        scaler = StandardScaler(
            inputCol="assembled_features",
            outputCol="features",
            withMean=True,
            withStd=True,
        ).fit(assembled_data)
        scaled_data = scaler.transform(assembled_data).select(
            *NUMERIC_COLUMNS, "features"
        )
        scaled_data.persist(StorageLevel.MEMORY_AND_DISK)

        scores = []
        all_correlations = []
        best_k = None
        best_score = float("-inf")
        for k in K_VALUES:
            model = KMeans(
                k=k,
                seed=42,
                maxIter=100,
                tol=1e-4,
                featuresCol="features",
                predictionCol="cluster",
            ).fit(scaled_data)
            clustered = model.transform(scaled_data).persist(
                StorageLevel.MEMORY_AND_DISK
            )
            try:
                scored_rows = clustered.select("features", "cluster").collect()
                feature_matrix = [
                    row["features"].toArray() for row in scored_rows
                ]
                labels = [row["cluster"] for row in scored_rows]
                score = float(silhouette_score(feature_matrix, labels))
                scores.append({"k": k, "silhouette": score})
                all_correlations.extend(correlation_rows(clustered, k))
                if score > best_score:
                    best_k = k
                    best_score = score
            finally:
                clustered.unpersist()

        if best_k is None:
            raise RuntimeError("Não foi possível selecionar o melhor K.")

        best_model = KMeans(
            k=best_k,
            seed=42,
            maxIter=100,
            tol=1e-4,
            featuresCol="features",
            predictionCol="cluster",
        ).fit(scaled_data)
        assignments = (
            best_model.transform(scaled_data)
            .select(*NUMERIC_COLUMNS, F.col("cluster").cast("int"))
            .toPandas()
        )
        write_outputs(
            pd.DataFrame(scores),
            pd.DataFrame(all_correlations),
            assignments,
            output_dir,
        )
    finally:
        if scaled_data is not None:
            scaled_data.unpersist()
        spark.stop()


def main() -> None:
    args = parse_args()
    run_analysis(args.input, Path(args.output))


if __name__ == "__main__":
    main()
