# Step 1: Use an official TensorFlow runtime as a parent image
FROM tensorflow/tensorflow:2.15.0

# Step 2: Set the working directory in the container
WORKDIR /app

# Step 3: Copy the current directory contents into the container at /app
COPY . /app

# Step 4: Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Step 5: Expose the port that the Flask app will run on
EXPOSE 8080

# Step 6: Define environment variables
ENV FLASK_APP=app.py

# Step 7: Run the application using Gunicorn
CMD ["gunicorn", "--bind", "0.0.0.0:8080", "app:app"]
