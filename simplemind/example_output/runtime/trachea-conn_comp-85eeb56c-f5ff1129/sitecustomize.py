import warnings
warnings.filterwarnings(
    'ignore',
    message=r'.*Applied workaround for CuDNN issue.*',
    category=UserWarning,
)
