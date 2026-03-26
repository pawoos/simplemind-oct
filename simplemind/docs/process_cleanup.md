# Process Cleanup

SM runs tools as subprocesses, that should terminate automatically when a run ends or the dashboard is closed.

If processes remain they may interact with new data posts, and so should be considered as a potential source of unexpected outputs.

To check whether there are active processes:
``` bash
pgrep -u "$USER" -af /smcore/
```
To terminate those processes:
``` bash
pkill -9 -u "$USER" -f /smcore/
```
