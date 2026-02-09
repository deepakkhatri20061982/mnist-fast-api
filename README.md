# Docker Commands
docker build -t mnist-fast-api .

docker run -d --name mnist-fast-api -p 8000:8000 -v mlruns2:/mlruns2 mnist-fast-api
