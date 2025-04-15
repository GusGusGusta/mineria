# 1. Uso una imagen base con Python ya instalado
FROM python:3.13

# 2. Creamos y seteamos el directorio de trabajo dentro del contenedor
WORKDIR /app

# 3. Copiamos todo el contenido del proyecto (incluyendo requirements.txt)
COPY . .

# 4. Instalamos dependencias desde requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# 5. Creamos la carpeta 'graficos' si no existe y actualizamos el sistema
RUN mkdir -p graficos && apt-get update && apt-get install -y \
    build-essential \
    libfreetype6-dev \
    libpng-dev \
    libjpeg-dev \
    fonts-dejavu \
    && rm -rf /var/lib/apt/lists/*

# Variables de entorno
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# 6. Comando por defecto para ejecutar el script principal
CMD ["python", "analisis.py"]
