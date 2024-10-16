FROM python:3.10

ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

WORKDIR /app

RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

RUN curl -fsSL https://ollama.com/install.sh | sh

COPY . /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

RUN echo '#!/bin/bash\nollama serve &\nsleep 5\npython interactive_classify.py' > /app/start.sh
RUN chmod +x /app/start.sh

CMD ["/app/start.sh"]