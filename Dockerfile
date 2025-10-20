FROM python:3.9-slim

# Set environment variable and prevent Python from wrting .pyc file and buffer outputs
ENV PYTHONDONTWITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set th working directory inside the container
WORKDIR /app

# Copy the requirements file to the container
COPY requirements.txt /app/requirements.txt

# Install the Python dependencies
RUN pip install -r /app/requirements.txt

# Copy the application code to the container
COPY . /app

# Expose the port the app runs on
EXPOSE 8000

# Command to run the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]