# Running on gpu-1 server

On the **gpu-1** server, a global BB is available.
* You go not need to start the core service
* The BB address is `bb-1.heph.com:8080`

To restart the BB on **bb-1** (rarely needed):
``` bash
sudo systemctl restart corebb@bb-1.heph.com:8080
```

Monitor messages:
``` bash
python -m smcore.listen --addr 127.0.0.1:8080
```
