FROM python:3.9-slim

WORKDIR /app

RUN groupadd -r appgroup && useradd -r -g appgroup appuser

USER appuser

COPY requirements.txt .

# Instalar as dependências
RUN pip install --no-cache-dir --user -r requirements.txt

# Copiar o restante dos arquivos da aplicação
COPY app/ .

# Definir o comando padrão para rodar o script
CMD ["python", "-m", "streamlit", "run", "main.py"]