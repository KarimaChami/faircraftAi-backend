#Utilise une image Linux qui contient Python 3.11
FROM python:3.11-slim 

#Empêche Python de créer __pycache__/*.pyc réduit la taille de l’image
ENV PYTHONDONTWRITEBYTECODE=1
#Force Python à afficher les logs immédiatement.
ENV PYTHONUNBUFFERED=1
#Définit le dossier de travail dans le container. / dossier principal de projet
WORKDIR /app
#Met à jour la liste des packages Linux. / Installe des outils système nécessaires./install PostgreSQL client libraries/Nettoie le cache après installation.lcach kayt9el limage 
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "src.app.main:app", "--host", "0.0.0.0", "--port", "8000"]
