FROM python:3.9-slim-bullseye as builder

RUN apt-get update && apt-get install -y \
    wget \
    gnupg \
    curl \
    unzip \
    && wget -q -O - https://dl-ssl.google.com/linux/linux_signing_key.pub | apt-key add - \
    && echo "deb [arch=amd64] http://dl.google.com/linux/chrome/deb/ stable main" >> /etc/apt/sources.list.d/google-chrome.list \
    && apt-get update \
    && apt-get install -y google-chrome-stable chromium-driver \
    && rm -rf /var/lib/apt/lists/*

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir torch torchvision --extra-index-url https://download.pytorch.org/whl/cpu

FROM python:3.9-slim-bullseye

RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libnss3 \
    libgconf-2-4 \
    libfontconfig1 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY --from=builder /usr/bin/google-chrome /usr/bin/google-chrome
COPY --from=builder /usr/lib/chromium /usr/lib/chromium

RUN groupadd -r amazonflex && useradd -r -g amazonflex -G audio,video amazonflex \
    && mkdir -p /home/amazonflex && chown -R amazonflex:amazonflex /home/amazonflex

RUN mkdir -p /app /models /logs /tmp/chrome
WORKDIR /app
COPY . .

RUN chown -R amazonflex:amazonflex /app /models /logs /tmp
USER amazonflex

ENV PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    MODEL_PATH=/models \
    LOG_PATH=/logs \
    TMPDIR=/tmp

EXPOSE 8080 9090

HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8080/healthz || exit 1

CMD ["gunicorn", "--bind", "0.0.0.0:8080", "--workers", "4", "--threads", "8", \
     "--timeout", "120", "--access-logfile", "-", "--error-logfile", "-", \
     "--preload", "app:app"]