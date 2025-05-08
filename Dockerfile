# Base image
FROM python:3.9-slim

# Working directory
WORKDIR /app

# Copy requirements first
COPY requirements.txt /app/

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . /app/

# Expose the port
EXPOSE 5001

# Run the application
CMD ["python", "./main.py"]