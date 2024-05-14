# DockerHubにあるpythonのイメージ指定する
FROM python:3.10-slim

RUN apt-get autoremove -y &&\
apt-get clean &&\
rm -rf /usr/local/src/*
# RUN apt update
# RUN apt indtall gcc
RUN apt-get update
RUN apt-get install gcc -y

# 作業ディレクトリをworkspaceに変更
WORKDIR /workspace/.devcontainer/web/

# ローカルPCのrequirements.txtをコンテナのカレントディレクトリ（上で指定した場所）にコピー
COPY requirements.txt requirements.txt

RUN pip install wheel setuptools pip --upgrade

# pipのアップデート
RUN pip install --upgrade pip

# pythonパッケージをインストール
RUN pip install --no-cache-dir -r requirements.txt

# CMD ["bin/sh"]