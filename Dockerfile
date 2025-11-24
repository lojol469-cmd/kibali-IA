FROM python:3.11-slim

WORKDIR /app

# Install system dependencies for osmium and other libraries
RUN apt-get update && apt-get install -y \
    libexpat1-dev \
    libboost-dev \
    libbz2-dev \
    liblz4-dev \
    libzstd-dev \
    cmake \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Expose the port Streamlit runs on
EXPOSE 8501

# Command to run the application
CMD ["streamlit", "run", "app.py", "--server.port", "8501", "--server.address", "0.0.0.0"]