FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Install uv
RUN pip install --no-cache-dir uv

# Copy dependency files
COPY pyproject.toml uv.lock ./

# Install project dependencies
RUN uv sync --frozen --no-dev --no-install-project

# Make the virtual environment available
ENV PATH="/app/.venv/bin:$PATH"

# Copy trained model
COPY models/ ./models/

# Copy API application
COPY src/fraud_detection/ ./src/fraud_detection/

# Expose API port
EXPOSE 8000

# Start FastAPI
CMD ["uvicorn", "main:app", "--app-dir", "src/fraud_detection", "--host", "0.0.0.0", "--port", "8000"]
