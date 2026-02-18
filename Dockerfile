# Start from the official lightweight Python base image
FROM python:3.11-slim

# Set the working directory inside the container
WORKDIR /app

COPY requirements.txt .

# THIS LINE IS CRITICAL - DO NOT DELETE IT
RUN pip install --no-cache-dir -r requirements.txt

# Now copy the rest of your application code into the container
COPY . .

# Expose the port that FastAPI will run on
EXPOSE 8000

# Set the command to run your FastAPI app using the Python module workaround
CMD ["python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]