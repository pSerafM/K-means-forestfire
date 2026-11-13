#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
HADOOP_VERSION="${HADOOP_VERSION:-3.5.0}"
HADOOP_HOME="${HADOOP_HOME:-$HOME/opt/hadoop-$HADOOP_VERSION}"
HDFS_DATA_DIR="${HDFS_DATA_DIR:-$HOME/.local/share/forestfire-hadoop}"
HDFS_URI="hdfs://localhost:9000"
HDFS_USER="${USER:?USER must be set}"
DOWNLOAD_DIR=""
STAGING_DIR=""
STARTED_NAMENODE=0
STARTED_DATANODE=0

cleanup() {
    local exit_code=$?
    if (( STARTED_DATANODE )); then
        "$HADOOP_HOME/bin/hdfs" --daemon stop datanode || true
    fi
    if (( STARTED_NAMENODE )); then
        "$HADOOP_HOME/bin/hdfs" --daemon stop namenode || true
    fi
    if [[ -n "$DOWNLOAD_DIR" ]]; then
        rm -rf -- "$DOWNLOAD_DIR"
    fi
    if [[ -n "$STAGING_DIR" && -d "$STAGING_DIR" ]]; then
        rm -rf -- "$STAGING_DIR"
    fi
    exit "$exit_code"
}
trap cleanup EXIT

daemon_is_running() {
    local status_output
    status_output="$(hdfs --daemon status "$1" 2>&1)" || return 1
    [[ "$status_output" == *"is running as process"* ]]
}

for command in curl java python3 readlink sha512sum tar; do
    if ! command -v "$command" >/dev/null 2>&1; then
        printf 'Required command not found: %s\n' "$command" >&2
        exit 1
    fi
done

JAVA_VERSION="$(java -version 2>&1 | head -n 1)"
if [[ ! "$JAVA_VERSION" =~ version\ \"17([.]|\\\") ]]; then
    printf 'Hadoop %s requires Java 17; found: %s\n' "$HADOOP_VERSION" "$JAVA_VERSION" >&2
    exit 1
fi
JAVA_HOME="${JAVA_HOME:-$(dirname -- "$(dirname -- "$(readlink -f -- "$(command -v java)")")")}"
export JAVA_HOME

if [[ ! -x "$HADOOP_HOME/bin/hdfs" ]]; then
    printf 'Hadoop %s is missing or incomplete; downloading a verified binary...\n' "$HADOOP_VERSION"
    DOWNLOAD_DIR="$(mktemp -d)"
    ARCHIVE="hadoop-$HADOOP_VERSION.tar.gz"
    BASE_URL="https://downloads.apache.org/hadoop/common/hadoop-$HADOOP_VERSION"
    curl --fail --location --retry 5 --retry-all-errors \
        --output "$DOWNLOAD_DIR/$ARCHIVE" "$BASE_URL/$ARCHIVE"
    curl --fail --location --retry 5 --retry-all-errors \
        --output "$DOWNLOAD_DIR/$ARCHIVE.sha512" "$BASE_URL/$ARCHIVE.sha512"

    EXPECTED_SHA512="$(awk '{print $NF}' "$DOWNLOAD_DIR/$ARCHIVE.sha512")"
    ACTUAL_SHA512="$(sha512sum "$DOWNLOAD_DIR/$ARCHIVE" | awk '{print $1}')"
    if [[ -z "$EXPECTED_SHA512" || "$EXPECTED_SHA512" != "$ACTUAL_SHA512" ]]; then
        printf 'SHA-512 verification failed for %s\n' "$ARCHIVE" >&2
        exit 1
    fi

    INSTALL_PARENT="$(dirname -- "$HADOOP_HOME")"
    mkdir -p -- "$INSTALL_PARENT"
    STAGING_DIR="$(mktemp -d "$INSTALL_PARENT/.hadoop-install.XXXXXX")"
    tar -xzf "$DOWNLOAD_DIR/$ARCHIVE" --strip-components=1 -C "$STAGING_DIR"
    if [[ ! -x "$STAGING_DIR/bin/hdfs" ]]; then
        printf 'The Hadoop archive did not contain bin/hdfs.\n' >&2
        exit 1
    fi

    if [[ -e "$HADOOP_HOME" ]]; then
        mv -- "$HADOOP_HOME" "$HADOOP_HOME.incomplete.$(date +%Y%m%d%H%M%S)"
    fi
    mv -- "$STAGING_DIR" "$HADOOP_HOME"
fi

export HADOOP_HOME
export HADOOP_CONF_DIR="$HADOOP_HOME/etc/hadoop"
export PATH="$HADOOP_HOME/bin:$HADOOP_HOME/sbin:$PATH"

cat > "$HADOOP_CONF_DIR/core-site.xml" <<'EOF'
<configuration>
    <property>
        <name>fs.defaultFS</name>
        <value>hdfs://localhost:9000</value>
    </property>
</configuration>
EOF

cat > "$HADOOP_CONF_DIR/hdfs-site.xml" <<EOF
<configuration>
    <property>
        <name>dfs.replication</name>
        <value>1</value>
    </property>
    <property>
        <name>dfs.namenode.name.dir</name>
        <value>file://${HDFS_DATA_DIR}/name</value>
    </property>
    <property>
        <name>dfs.datanode.data.dir</name>
        <value>file://${HDFS_DATA_DIR}/data</value>
    </property>
</configuration>
EOF

mkdir -p "$HDFS_DATA_DIR/name" "$HDFS_DATA_DIR/data"
sed -i '/^[[:space:]]*export JAVA_HOME=/d' "$HADOOP_CONF_DIR/hadoop-env.sh"
printf 'export JAVA_HOME="%s"\n' "$JAVA_HOME" >> "$HADOOP_CONF_DIR/hadoop-env.sh"

if [[ ! -f "$HDFS_DATA_DIR/name/current/VERSION" ]]; then
    printf 'Initializing the local HDFS NameNode for its first use...\n'
    hdfs namenode -format -nonInteractive
fi

if ! daemon_is_running namenode; then
    hdfs --daemon start namenode
    STARTED_NAMENODE=1
fi
if ! daemon_is_running datanode; then
    hdfs --daemon start datanode
    STARTED_DATANODE=1
fi

HDFS_READY=0
for _ in {1..90}; do
    if curl --fail --silent --show-error --max-time 2 \
        http://localhost:9870/ >/dev/null 2>&1; then
        HDFS_READY=1
        break
    fi
    sleep 1
done
if (( ! HDFS_READY )); then
    printf 'NameNode web service did not become ready. Check Hadoop logs under %s/logs.\n' "$HADOOP_HOME" >&2
    for log_file in "$HADOOP_HOME"/logs/*namenode*.log "$HADOOP_HOME"/logs/*datanode*.log; do
        if [[ -f "$log_file" ]]; then
            printf '\n--- Last lines of %s ---\n' "$log_file" >&2
            tail -n 40 "$log_file" >&2
        fi
    done
    exit 1
fi
if ! hdfs dfs -ls /; then
    printf 'HDFS NameNode is reachable but the filesystem command failed.\n' >&2
    exit 1
fi

if [[ ! -f "$ROOT_DIR/.venv/bin/activate" ]]; then
    python3 -m venv "$ROOT_DIR/.venv"
fi
source "$ROOT_DIR/.venv/bin/activate"
if ! python -c 'import matplotlib, pandas, pyspark, sklearn' >/dev/null 2>&1; then
    python -m pip install -r "$ROOT_DIR/requirements.txt"
fi

hdfs dfs -mkdir -p "/user/$HDFS_USER/forestfire"
hdfs dfs -put -f "$ROOT_DIR/forestfireinput.csv" \
    "/user/$HDFS_USER/forestfire/forestfireinput.csv"

spark-submit \
    --master 'local[*]' \
    "$ROOT_DIR/src/main.py" \
    --input "$HDFS_URI/user/$HDFS_USER/forestfire/forestfireinput.csv" \
    --output "$ROOT_DIR/resultados_kmeans"
