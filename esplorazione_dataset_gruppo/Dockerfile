FROM python:3.12
WORKDIR /app
COPY requiremets.txt .
RUN pip install --no-cache-dir -r requiremets.txt
COPY . .
EXPOSE 5000
CMD ["python", "-m", "main"]