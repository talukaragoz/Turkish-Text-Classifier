FROM python:3.10

WORKDIR /app

RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && curl -fsSL https://ollama.com/install.sh | sh

COPY requirements.txt .
COPY interactive_classify.py .
COPY llm_query.py .
COPY model_query.py .
COPY models/ ./models/

RUN pip install --no-cache-dir -r requirements.txt

RUN echo '#!/bin/bash\nollama serve &\nsleep 5\npython interactive_classify.py' > /app/start.sh \
    && chmod +x /app/start.sh

CMD ["/app/start.sh"]
