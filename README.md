docker build -t mnist-fast-api .

docker run -d --name mnist-fast-api -p 8000:8000 mnist-fast-api
