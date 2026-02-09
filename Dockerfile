FROM python:3.10-slim

WORKDIR /app

# Copia tutto il progetto
COPY . .

# Installa le dipendenze
RUN pip install --no-cache-dir torch==2.2.0+cpu torchvision==0.17.0+cpu --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir numpy==1.26.4 pandas matplotlib scikit-learn

# Esegui un comando di default (puoi cambiarlo)
CMD ["python", "-m", myproject.predict"]

