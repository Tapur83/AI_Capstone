FROM python:3.9-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY *.py ./
COPY cs-train/ ./cs-train/
COPY cs-production/ ./cs-production/

# Create directories for models, logs, figures
RUN mkdir -p models logs figures

# Expose port
EXPOSE 8080

# Train production models on startup and launch Flask API
CMD python -c "from model import model_train; model_train('cs-train', test=False)" && python -m flask --app app run --host=0.0.0.0 --port=8080
