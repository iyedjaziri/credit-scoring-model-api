# Use python 3.12-slim to match development environment
FROM python:3.12-slim

# Install system dependencies required for LightGBM and SHAP
RUN apt-get update && apt-get install -y \
    libgomp1 \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Install poetry and export plugin
RUN pip install poetry poetry-plugin-export

# Copy project definition
COPY pyproject.toml poetry.lock ./

# Export dependencies to requirements.txt and install
# This avoids keeping poetry in the final image and ensures a clean install
RUN poetry export -f requirements.txt --output requirements.txt --without-hashes \
    && pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY src/ src/
COPY model/ model/

# Expose port
EXPOSE 8000

# Command to run the application
# We use src.api.main:app because the code is in src/api/main.py and PYTHONPATH should include /app (default in current workdir)
ENV PYTHONPATH=/app
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
