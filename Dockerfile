# Use an official Python runtime as a parent image
FROM python:3.10.9-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set the working directory
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt /app/

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code into the container
COPY . /app/

# Optional: Add a non-root user
RUN useradd -ms /bin/sh appuser
USER appuser

# Expose the port that the application will run on
EXPOSE 8080

# Command to run the application
CMD ["gunicorn", "-b", "0.0.0.0:8080", "-w", "4", "--timeout", "120", "app:app"]
