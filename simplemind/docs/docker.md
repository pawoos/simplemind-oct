# Docker Environment

(for only maintainer) Build and push the container:
``` bash
docker build -t cvib/sm-incubator -f Dockerfile .
docker push cvib/sm-incubator:latest
```

Run a local Core service:
``` bash
docker run -p 8080:8080 -it cvib/core bash
```
If using a local Core, replace `bb-1.heph.com:8080` with `127.0.0.1:8080` in commands.

If using Docker, spin up a container and run subsequent commands inside it (update volume mount as needed):
``` bash
docker run --gpus all --shm-size=1G --network=host -e ENV_NAME=smcore -it -u $(id -u):$(id -g) -w $PWD -v $PWD:$PWD cvib/sm-incubator bash
```
