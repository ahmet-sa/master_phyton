# Use the official TensorFlow Docker image as a base
FROM tensorflow/tensorflow:2.11.0-py3

# Install Flask
RUN pip install Flask

# Set up working directory
WORKDIR /app

# Copy your application code to the container
COPY . /app

# Expose the port Flask is running on
EXPOSE 8080

# Run the Flask application
CMD ["flask", "run", "--host=0.0.0.0", "--port=8080"]
