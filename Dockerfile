FROM python:3.9-slim

COPY requirements.txt .

# Instalar as dependências
RUN pip install --no-cache-dir -r requirements.txt

WORKDIR /appml

RUN groupadd -r appuser && useradd -r -g appuser appuser

RUN mkdir -p /home/appuser/.streamlit && \
    chown -R appuser:appuser /home/appuser /appml

ENV STREAMLIT_HOME=/home/appuser/.streamlit

USER appuser

CMD ["python", "--version"]