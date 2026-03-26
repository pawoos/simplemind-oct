from __future__ import annotations
import asyncio
from pathlib import Path

from sm_sample_processor import SMSampleProcessor
from sm_image import SMImage
from sm_sample_id import SMSampleID
from env_helper import setup_env, call_in_env

# Allow overrides
# ENV_NAME = os.environ.get("TOTALSEG_ENV_NAME", "totalseg")
# ENV_NAME = "totalseg"
# LIB_BASE_DIR = Path("../../../lib")
# LIB_DIR = Path("../../../lib") / ENV_NAME

...
class TotalSeg(SMSampleProcessor):
    async def execute(
        self,
        *,
        input_image: SMImage,
        sample_id: SMSampleID,
        task: str = "total",
        fast: bool = False,
        output_mode: str = "labelmap_raw",
        roi_subset: str | None = None,
        output_dir: str = "./samples",   
    ) -> SMImage:


        if input_image is None:
            return None

        args = ["--task", task, "--outdir", output_dir, "--sample_id", str(sample_id),
                "--output_mode", output_mode]
        if fast:
            args.append("--fast")
        if roi_subset:
            args += ["--roi_subset", roi_subset]

        result_bytes = call_in_env(
            script_name="ts_main.py",
            input_data=input_image.to_bytes(),
            script_args=args,
            env_name="totalseg",
            hash_dir=Path("../../../lib"),
            setup_env_func=setup_env
        )
        return SMImage.from_bytes(result_bytes)

if __name__ == "__main__":
    tool = TotalSeg()
    asyncio.run(tool.main())
