# 1. Base image
FROM python:3.10

# 2. Copy files
COPY . /src

# 3. Install dependencies
RUN pip install -r /src/requirements.txt

# 4. Run the application
CMD ["python", "/src/googletrans_llava.py"]
