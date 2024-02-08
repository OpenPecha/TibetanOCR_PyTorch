import albumentations as A



train_transform = A.Compose(
[
        #A.ShiftScaleRotate(shift_limit=0.02, scale_limit=0.01, rotate_limit=1, p=0.5), 
        A.Affine(scale=None, translate_percent=None, translate_px=None, rotate=[0, 0], shear={"x": (-10, 10), "y": (-0.1, 0.1)}, interpolation=1, mask_interpolation=0, cval=0, cval_mask=0, mode=0, fit_output=False, keep_ratio=False, rotate_method='largest_box', always_apply=True, p=0.5),
        #A.RandomBrightnessContrast(p=0.1),
        #A.Downscale(scale_min=0.5, scale_max=0.5, interpolation=None, always_apply=False, p=0.5),
        A.Emboss (alpha=(0.2, 0.5), strength=(0.2, 0.7), always_apply=False, p=0.5),
        A.GaussNoise(var_limit=(10.0, 50.0), mean=0, per_channel=True, always_apply=False, p=0.5),
        A.PixelDropout(dropout_prob=0.02, per_channel=False, drop_value=0, mask_drop_value=None, always_apply=False, p=0.5),
        A.RandomGravel (gravel_roi=(0.2, 0.4, 0.9, 0.9), number_of_patches=2, always_apply=False, p=0.5),
        #A.RandomRain (slant_lower=-10, slant_upper=10, drop_length=20, drop_width=1, drop_color=(200, 200, 200), blur_value=2, brightness_coefficient=0.7, rain_type=None, always_apply=False, p=0.5)
        A.AdvancedBlur (blur_limit=(3, 5), sigmaX_limit=(0.2, 1.0), sigmaY_limit=(0.2, 1.0), rotate_limit=90, beta_limit=(0.5, 8.0), noise_limit=(0.9, 1.1), always_apply=False, p=0.5)
        
    ]
)