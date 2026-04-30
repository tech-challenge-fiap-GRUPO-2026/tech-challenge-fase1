FROM python:3.9-slim AS treinamento

COPY requirements.txt .

# Instalar as dependências
RUN pip install --no-cache-dir -r requirements.txt

WORKDIR /appml

RUN groupadd -r appuser && useradd -r -g appuser appuser

RUN mkdir -p /home/appuser/.streamlit && \
    chown -R appuser:appuser /home/appuser /appml

USER appuser

COPY --chown=appuser:appuser . .

RUN /bin/bash run_training.sh


FROM python:3.9-slim

COPY requirements.txt .

# Instalar as dependências
RUN pip install --no-cache-dir -r requirements.txt

WORKDIR /appml

RUN groupadd -r appuser && useradd -r -g appuser appuser

RUN mkdir -p /home/appuser/.streamlit && \
    chown -R appuser:appuser /home/appuser /appml

USER appuser

COPY --chown=appuser:appuser . .

COPY --from=treinamento /appml/model/ /appml/model/

CMD ["/bin/bash", "run_api.sh"]