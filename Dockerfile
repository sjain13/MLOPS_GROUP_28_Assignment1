# Use an official Python runtime as the base image
FROM python:3.10.0

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set the working directory in the container
WORKDIR /app

# Copy only the requirements first to leverage Docker caching
COPY requirements.txt /app/requirements.txt

# Install dependencies
RUN pip install --no-cache-dir -r /app/requirements.txt


# Copy the rest of the application code
COPY . /app

# Copy entrypoint script
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

EXPOSE 5000 5001

# Run entrypoint script
ENTRYPOINT ["/entrypoint.sh"]

# Command to run the application
#CMD ["python", "src/predictdiabetes_app.py"]
