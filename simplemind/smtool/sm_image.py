import numpy as np
import numpy.typing as npt
import base64
import json
import os
from collections import OrderedDict
import matplotlib
import matplotlib.pyplot as plt
if os.environ.get('DISPLAY','') == '':
    # print('No display found. Using non-interactive Agg backend.')
    matplotlib.use('Agg')
import matplotlib.colors as mcolors

from smcore import serialize, deserialize

class SMImage:
    """
    An image container class for use with SimpleMind tools.
    SMImage is intended to help standardize access patterns to common structured imaging data often found in medical imaging.

    SMImage is not intended to provide image manipulation functionality such as pre or postprocessing, which is intentionally delegated
    to the agents that will utilize SMImages.
    SMImage should be understood to be permissive.  We favor returning empty values or error values, rather than requiring strict adherence to
    specific metadata patterns.
    
    SMImage stores pixel_array and label_array with dimensions: [C, Z, Y, X]
    - C is channels (typically 1)
    - For 2D, Z = 1
    """
    
    metadata: dict                                  # image header information based on nifti format
    #pixel_array: npt.NDArray[np.int_ | np.float_]   # numpy array of int or float, i.e., image or mask
    pixel_array: npt.NDArray[np.int_ | np.float64]   # numpy array of int or float, i.e., image or mask
    label_array: npt.NDArray[np.int_]               # numpy array of int, i.e., label mask(s)

    sm_image_tag = "sm_image-data"

    def __init__(self, metadata: dict, pixel_array: np.ndarray, label_array: np.ndarray = None):
        if pixel_array is None:
            raise ValueError("sm_image: pixel_array cannot be None")
        
        self.metadata = metadata
        # self.pixel_array = pixel_array 
        # self.label_array = label_array
        self.pixel_array = SMImage.normalize_dims(pixel_array)
        if label_array is not None:
            self.label_array = SMImage.normalize_dims(label_array)
        else:
            self.label_array = None
            

    def spacing(self) -> tuple:
        """Returns the spacing of the image with numpy ordering: (z, y, x)."""
        if "spacing" in self.metadata:
            return tuple(self.metadata["spacing"][::-1])
        return None

    def origin(self) -> tuple:
        """Returns the origin of the image with numpy ordering: (z, y, x)."""
        if "origin" in self.metadata:
            return tuple(self.metadata["origin"][::-1])
        return None

    @classmethod
    def tag(cls) -> str:
        """Return the canonical tag for an SMImage post"""
        return cls.sm_image_tag

    @classmethod
    def _test_image(cls):
        metadata = {"origin": (0, 0, 0), "spacing": (0.5, 0.5, 0.5)}
        image = np.random.rand(1, 512, 512)
        label = None
        return cls(metadata, image, label)
    
    def to_bytes(self) -> bytes:
        pixel_bytes = serialize.numpy(self.pixel_array)
        label_bytes = serialize.compressed_numpy(self.label_array) if self.label_array is not None else b''
        data = {
            'pixel_array': base64.b64encode(pixel_bytes).decode('utf-8'),
            'label_array': base64.b64encode(label_bytes).decode('utf-8') if label_bytes else '',
            'metadata': self.metadata,
        }
        return json.dumps(data).encode('utf-8')
    
    @classmethod
    def from_bytes(cls, data: bytes):
        # Decode JSON
        obj = json.loads(data.decode('utf-8'))

        # Decode pixel_array
        pixel_bytes = base64.b64decode(obj['pixel_array'])
        pixel_array = deserialize.numpy(pixel_bytes)  # assuming serialize.from_numpy reverses serialize.numpy

        # Decode label_array if present
        label_array = None
        if obj.get('label_array'):
            label_bytes = base64.b64decode(obj['label_array'])
            label_array = deserialize.compressed_numpy(label_bytes)  # reverse of compressed_numpy

        metadata = obj.get('metadata', {})

        return cls(pixel_array=pixel_array, label_array=label_array, metadata=metadata)

    @staticmethod
    def normalize_dims(arr: np.ndarray) -> np.ndarray:
        """
        Ensures array shape follows channel-first convention: (1, D, H, W)
        """
        if arr.ndim == 2:
            # (H, W) -> (1, H, W)
            return arr[np.newaxis, np.newaxis, :, :]
        elif arr.ndim == 3:
            return arr[np.newaxis, :, :, :]
        elif arr.ndim == 4:
            # (1, D, H, W) -> unchanged
            return arr
        else:
            raise ValueError(f"Unsupported array shape {arr.shape}; expected 2 - 4D input.")
        
# cmap_dict = {'pink': matplotlib.cm.spring,
#                 'green': matplotlib.cm.summer,
#                 'torquise': matplotlib.cm.cool,
#                 'cyan':matplotlib.cm.cool,
#                 'blue': matplotlib.cm.winter,
#                 'brightGreen': matplotlib.cm.winter_r,
#                 'orange': matplotlib.cm.Wistia_r,
#                 'red': matplotlib.cm.autumn,
#                 'yellow': matplotlib.cm.autumn_r,
#                 'purple': matplotlib.cm.PRGn,
#                 'magenta': matplotlib.cm.PiYG,
#                 'parrotGreen': matplotlib.cm.PiYG_r,
#                 'b_green': matplotlib.cm.brg_r,
#                 'yellow_green': matplotlib.cm.Wistia,
#                 'light_blue': matplotlib.cm.ocean_r}

cmap_dict = {
    'pink': mcolors.ListedColormap([[1.0, 0.75, 0.8, 1]]),
    'green': mcolors.ListedColormap([[0.0, 1.0, 0.0, 1]]),
    'torquise': mcolors.ListedColormap([[0.25, 0.88, 0.82, 1]]),
    'cyan': mcolors.ListedColormap([[0.0, 1.0, 1.0, 1]]),
    'blue': mcolors.ListedColormap([[0.0, 0.0, 1.0, 1]]),
    'brightGreen': mcolors.ListedColormap([[0.5, 1.0, 0.0, 1]]),
    'orange': mcolors.ListedColormap([[1.0, 0.65, 0.0, 1]]),
    'red': mcolors.ListedColormap([[1.0, 0.0, 0.0, 1]]),
    'yellow': mcolors.ListedColormap([[1.0, 1.0, 0.0, 1]]),
    'purple': mcolors.ListedColormap([[0.5, 0.0, 0.5, 1]]),
    'magenta': mcolors.ListedColormap([[1.0, 0.0, 1.0, 1]]),
    'parrotGreen': mcolors.ListedColormap([[0.19, 0.80, 0.19, 1]]),
    'b_green': mcolors.ListedColormap([[0.0, 0.5, 0.0, 1]]),
    'yellow_green': mcolors.ListedColormap([[0.6, 0.8, 0.2, 1]]),
    'light_blue': mcolors.ListedColormap([[0.53, 0.81, 0.98, 1]]),
}
    
def view_image(arr, save_path, spacing = [1,1,1], mask=None, alpha=0.8,mask_type=None,mask_name=None,mask_color=None):
    logs = ""

    fill_color = None
    if mask_color is not None and mask_color in list(cmap_dict.keys()):
        fill_color = cmap_dict[mask_color]
        if mask_type is not None:
            fill_color = mask_color
    else:
        fill_color='autumn'
        if mask_type is not None:
            fill_color = 'red'

    arr = np.squeeze(arr) # Squeeze the channel axis
    if mask is not None:
        mask = np.squeeze(mask) # Squeeze the channel axis
    # NOTE will fail for multichannel preprocessed images, need to account for that
    # TODO maybe another debug function for the preprocessor and not touch the preprocessed image here

    if arr.ndim < 3: # For 2D images
        logs +=  "    For 2D images    "

        arr = np.expand_dims(arr, axis = 0)
        if mask is not None:
             mask = np.expand_dims(mask, axis = 0)

        plt.figure(0, figsize=(20,20))
        if mask_name is not None:

            plt.suptitle(mask_name, fontsize=30)
        plt.imshow(arr[0], cmap = 'gray')
        if mask is not None:

            # if mask_type is not None:
            if mask_type is not None:
                if 'NA' not in mask:
                    # plt.plot(mask['xy'][0], mask['xy'][1], marker='x', color='red', 
                    #         linestyle = 'None', label=f'{mask_type} prediction')
                    # marker_style = dict(color='tab:red', linestyle=':', marker='x', markersize=15, markerfacecoloralt='tab:red')
                    logs +=  "    Here 1    "
                    plt.plot(mask[0], mask[1], marker='X',markersize=20, color=fill_color, interpolation='none',
                                            linestyle = 'None', label=f'{mask_type} prediction')
                    # plt.plot(mask[0], mask[1], marker='x',markersize=20,linewidth=3, color='red', 
                    #                         linestyle = 'None', label=f'{mask_type} prediction')
            else:
                #pass
                # mask = (mask > 0.5).astype(np.uint8)
                # print("view_image", arr.shape, mask.shape, mask.dtype, np.unique(mask), flush=True)
                plt.imshow(np.ma.masked_where(mask[0] == 0, mask[0]), cmap = fill_color, alpha=alpha, interpolation='none')
                #plt.imshow(np.ma.masked_where(mask == 0, mask), cmap = fill_color, alpha=alpha)
    
    else: # 3D images
        logs +=  "    For 3D images    "

        # percents = np.array([0.10, 0.25, 0.50, 0.75, 0.90]) # Can be replaced with linspace
        # Would arange going by step be better?
        # percents = np.linspace(0, 1, 10, endpoint=False) # This last integer will determine the amount of "slices" to show
        percents = np.arange(0, 1, 0.15)

        slices = np.floor(percents*arr.shape[0]).astype(int)
        coronal = np.floor(percents*arr.shape[1]).astype(int)
        sagittal = np.floor(percents*arr.shape[2]).astype(int)

        slices = list(OrderedDict.fromkeys(slices)) # Is this really needed anymore?
        coronal = list(OrderedDict.fromkeys(coronal))
        sagittal = list(OrderedDict.fromkeys(sagittal))
        plt.figure(0, figsize=(len(slices)*10,30))

        ss = spacing[2] # Assuming x, y, z format
        ps = spacing[1]

        cor_aspect = ss/ps
        for idx, z in enumerate(slices):

            sp1 = plt.subplot(3, len(percents), idx + 1)
            plt.imshow(arr[z], cmap = 'gray', interpolation='none')
            # plt.gca().set_title('title')
            if mask is not None:
                plt.imshow(np.ma.masked_where(mask[z, :,:] == 0, mask[z,:,:]), cmap = fill_color, alpha=alpha, interpolation='none')
                #m2d = np.ma.masked_where(mask[z, :,:] == 0, mask[z,:,:])
                #if m2d.ndim==2:
                #    plt.imshow(m2d, cmap = fill_color, alpha=0.8)
            sp1.set_title(f'{np.round(percents[idx]*100)}%', fontsize=25)
        
        for y in coronal:
            idx = idx + 1
            sp2 = plt.subplot(3, len(percents), idx + 1)
            plt.imshow(arr[:,y,:], cmap = 'gray', interpolation='none')
            if mask is not None:
                plt.imshow(np.ma.masked_where(mask[:,y,:] == 0, mask[:,y,:]), cmap = fill_color, alpha=alpha, interpolation='none')
            sp2.set_aspect(cor_aspect)

        for x in sagittal:
            idx = idx + 1 
            sp3 = plt.subplot(3, len(percents), idx + 1)
            plt.imshow(arr[:,:, x], cmap = 'gray', interpolation='none')
            if mask is not None:
                plt.imshow(np.ma.masked_where(mask[:,:, x] == 0, mask[:,:, x]), cmap = fill_color, alpha=alpha, interpolation='none')
            sp3.set_aspect(cor_aspect) # Will this aspect ratio work for sagittal?

    plt.savefig(save_path)
    plt.close()
    return logs

