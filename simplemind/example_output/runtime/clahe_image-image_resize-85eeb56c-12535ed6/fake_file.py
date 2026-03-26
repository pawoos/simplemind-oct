import os
import io
import tempfile
import shutil


# import sys
# import SimpleITK as sitk


# TODO: figure out a way that we' don't have to go out to
# disk to accomodate SITK. os.mkfifo seemed promising but
# I wasn't able to get it to work.
class FakeFile:
    def __init__(self, fileobj: io.BytesIO, ext: str = None):
        self.fileobj = fileobj
        self.temp_dir = tempfile.TemporaryDirectory()
        filename = "fakefile" + ext if ext is not None else "fakefile"
        self.fp = os.path.join(self.temp_dir.name, filename)

    def __enter__(self):
        with open(self.fp, "wb") as f:
            shutil.copyfileobj(self.fileobj, f)
        return self.fp

    # Clean up tempfile on exit
    def __exit__(self, exc_type, exc_value, traceback):
        shutil.rmtree(self.temp_dir.name)


# if __name__ == "__main__":
#     input = sys.argv[1]
#     with open(input, "rb") as f:
#         data = io.BytesIO(f.read())
#
#     with FakeFile(data, ext=".nii.gz") as ff:
#         test = sitk.ReadImage(ff)
#
#     print(test)
