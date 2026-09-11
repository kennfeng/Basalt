FROM python:3.12-slim
ENV PIP_NO_CACHE_DIR=1 PYTHONUNBUFFERED=1 HF_HOME=/data/hf-cache
WORKDIR /app
COPY requirements.txt .
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch==2.12.1 && pip install -r requirements.txt
COPY . .
RUN useradd -m -u 10001 basalt && mkdir -p /data/hf-cache /data/basalt_db && chown -R basalt:basalt /app /data && chmod +x scripts/entrypoint.sh
USER basalt
EXPOSE 8000
ENTRYPOINT ["/bin/sh", "scripts/entrypoint.sh"]
