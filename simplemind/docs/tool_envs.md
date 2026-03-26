# Tool Environments

How to create a tool that runs in its own environment, using the [`medsam2`](../tools/neural_net/medsam2/medsam2.py) tool as an example.

***

## Using `call_in_env`

As in `medsam2.execute`, the `call_in_env` function is what you will use to run your tool's `_main_` module.
``` python
    result_bytes = call_in_env(
        script_name="ms2.py",
        input_data=input_image.to_bytes(),
        script_args=["--z_index", str(z_index), "--prompt", str(prompt[0]), str(prompt[1]), str(prompt[2]), str(prompt[3])],
        env_name=ENV_NAME,
        lib_dir=LIB_DIR,
        setup_env_func=setup_env
    )
```
Parameters:
* **script_name**: your tool's `_main_` module, e.g., [`ms2`](../tools/neural_net/medsam2/ms2.py)
* **input_data**: SMImage data passed to `_main_` via `stdin` as bytes
    * encoded using `SMImage.to_bytes()`
    * decoded in [ms2](../tools/neural_net/medsam2/ms2.py) using `SMImage.from_bytes()`
* **script_args**: parsed by `argparse` in `_main_`
* **env_name**: the tool's micromamba environment
    * provide an [`env.yaml`](../tools/neural_net/medsam2/env.yaml)
    * the environment is automatically created during tool setup (the first time it runs)
* **lib_dir**: the path to the tool library
    * downloaded automatically during tool setup from a git repo (`url` in [`env.yaml`](../tools/neural_net/medsam2/env.yaml))
    * must match `dest` in [`env.yaml`](../tools/neural_net/medsam2/env.yaml)
* **setup_env_func**: use the tool setup function from [`env_helper`](../smtool/env_helper.py)

Returns:
* `SMImage` mask as a byte stream
    * encoded in [`ms2`](../tools/neural_net/medsam2/ms2.py) and decoded in [`medsam2`](../tools/neural_net/medsam2/medsam2.py) as above


***

## Running the medsam2 example

``` bash
python run_plan.py plans/medsam2 --dataset_csv data/ct_lesion.csv --addr bb-1.heph.com:8080
```
Clean up:
``` bash
rm -r working-*
```
``` bash
rm -r -f lib/MedSAM2
```