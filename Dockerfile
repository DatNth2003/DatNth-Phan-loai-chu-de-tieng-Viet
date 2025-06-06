FROM python:3.11-slim

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    default-jre-headless \
    && apt-get clean && \
    rm -rf /var/lib/apt/lists/*

ENV JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64
ENV PATH="${JAVA_HOME}/bin:${PATH}"

ENV APP_USER=appuser
RUN useradd -ms /bin/bash ${APP_USER}

ENV HF_HOME=/home/${APP_USER}/.cache/huggingface
ENV MPLCONFIGDIR=/home/${APP_USER}/.config/matplotlib

WORKDIR /app

COPY --chown=${APP_USER}:${APP_USER} requirements.txt ./
COPY --chown=${APP_USER}:${APP_USER} vncorenlp_lib ./vncorenlp_lib/
COPY --chown=${APP_USER}:${APP_USER} stopwords_tokenized.txt ./
COPY --chown=${APP_USER}:${APP_USER} static/ ./static/
COPY --chown=${APP_USER}:${APP_USER} templates ./templates/
COPY --chown=${APP_USER}:${APP_USER} app.py ./

RUN pip install --no-cache-dir -r requirements.txt

USER ${APP_USER}

EXPOSE 7860

CMD ["gunicorn", "app:app", "--bind", "0.0.0.0:7860", "--workers", "2", "--threads", "2", "--timeout", "120", "--preload"]