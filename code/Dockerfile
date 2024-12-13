# Base image
FROM python:3.9-slim

# Set the working directory
WORKDIR /app

# Copy application files to the container
COPY . .

# Install required Python libraries
RUN pip install --no-cache-dir -r requirements.txt

# Expose Streamlit's default port
EXPOSE 8501

# Command to run the app
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
