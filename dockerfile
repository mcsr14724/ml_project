FROM python:3.11-slim-bookworm

WORKDIR /app

COPY . /app

# Install awscli safely
RUN apt-get update && \
    apt-get install -y awscli && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 5000

CMD ["python", "application.py"]