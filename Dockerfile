# Use official Python slim image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy project files
COPY . /app

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Set environment variable for port
ENV PORT=8080

# Start the app with Gunicorn
CMD ["gunicorn", "--bind", ":8080", "--workers", "1", "--threads", "8", "app:app"]
