from pydantic import BaseModel, Field, field_validator, model_validator
from typing import Any
from os.path import exists
from dataclasses import dataclass


@dataclass(frozen=True, init=False)
class CONSTANTS:
    SUPPORTED_FILE_EXTENSIONS: tuple[str] = (".tif", ".tiff", ".h5", ".hdf5")


class DataParameters(BaseModel):
    """
    General params describing the dataset like dimensions, decay time, filename and framerate

    Attributes
    ----------
    fnames: list of str
        list of complete paths to files that need to be processed

    dims: tuple of int
        dimension of the FOV in pixels

    fr: float
        imaging frame rate in frames per second

    decay_time: float
        length of a typical transient in seconds

    dxy: tuple of float
        spatial resolution of FOV in pixels per um

    var_name_hdf5: str
        if loading from hdf5 name of the variable to load

    caiman_version: str
        version of CaImAn being used. Please do not override this

    last_commit: str
        hash of last commit in the caiman repo. Please do not override this.
    """

    #: list of complete paths to files that need to be processed
    fnames: list[str]

    #: dimension of the FOV in pixels
    dims: tuple[int, ...]

    #: imaging frame rate in frames per second.
    fr: float = Field(default=30.0, gt=0.0)

    #: length of a typical transient in seconds.
    decay_time: float = Field(default=1.0, gt=0.0)

    #: spatial resolution of FOV in pixels per um
    dxy: tuple[float, ...] = (1.0, 1.0)

    #: if loading from hdf5 name of the variable to load
    var_name_hdf5: str | None = None

    #: version of CaImAn being used. Please do not override this
    caiman_version: str = "1.0.0"

    #: hash of last commit in the caiman repo. Please do not override this.
    last_commit: str = "1234567890"

    @field_validator("fnames", mode="before")
    @classmethod
    def validate_fnames(cls, fnames: Any) -> list[str]:
        """
        Validates fnames is a list of strings whose length is greater than 0,
        whose members all exist, and whose file extensions indicate a supported
        format.

        Parameters
        ----------
        fnames: Potential list of strings to validate

        Returns
        -------
        list of str
            list of complete paths to files that need to be processed
        """
        # ensure fnames is a list
        if isinstance(fnames, list):
            pass
        elif isinstance(fnames, tuple | set):
            fnames = list(fnames)
        else:
            fnames = [str(fnames), ]

        # ensure all members are strings
        fnames = [str(fname) for fname in fnames]

        # ensure all members exist
        assert all([exists(fname) for fname in fnames]), "All members of fnames must exist"

        # ensure all members have a supported file extension
        assert(all([fname.endswith(CONSTANTS.SUPPORTED_FILE_EXTENSIONS) for fname in fnames]))

        return fnames

    @field_validator("dims", mode="after")
    @classmethod
    def validate_nonzero_dims(cls, dims: tuple[int, ...]) -> tuple[int, ...]:
        """
        Validates dimensions are greater than 0

        Parameters
        ----------
        dims: tuple of int
            dimension of the FOV in pixels

        Returns
        -------
        tuple of int
            dimension of the FOV in pixels

        Raises
        ------
        AssertionError
            if any value in dims is less than or equal to 0
        """
        assert all([value > 0 for value in dims]), "All values in dims must be greater than 0"
        return dims

    @field_validator("dxy", mode="after")
    @classmethod
    def validate_nonzero_dxy(cls, dxy: tuple[float, ...]) -> tuple[float, ...]:
        """
        Validates dxy are greater than 0

        Parameters
        ----------
        dxy: tuple of float
            spatial resolution of FOV in pixels per um

        Returns
        -------
        tuple of float
            spatial resolution of FOV in pixels per um

        Raises
        ------
        AssertionError
            if any value in dxy is less than or equal to 0
        """
        assert all([value > 0.0 for value in dxy]), "All values in dxy must be greater than 0"
        return dxy

    @model_validator(mode="after")
    def validate_movies_consistent(self):
        ...

    @model_validator(mode="after")
    def validate_dims_consistent_with_movie(self):
        ...

    @model_validator(mode="after")
    def validate_dxy_consistent_with_movie(self):
        ...


# noinspection DuplicatedCode
class PatchParameters(BaseModel):
    border_pix: Any
    del_duplicates: Any
    in_memory: Any
    low_rank_background: Any
    memory_fact: Any
    n_processes: Any
    nb_patch: Any
    only_init: Any
    p_patch: Any
    remove_very_bad_comps: Any
    rf: Any
    skip_refinement: Any
    p_ssub: Any
    stride: Any
    p_tsub: Any


class PreprocessParameters(BaseModel):
    check_nan: Any
    compute_g: Any
    include_noise: Any
    lags: Any
    max_num_samples_fft: Any
    n_pixels_per_process: Any
    noise_method: Any
    noise_range: Any
    p: Any
    pixels: Any
    sn: Any


class InitParameters(BaseModel):
    K: Any
    SC_kernel: Any
    SC_sigma: Any
    SC_thr: Any
    SC_normalize: Any
    SC_use_NN: Any
    SC_nnn: Any
    alpha_snmf: Any
    center_psf: Any
    gSig: Any
    gSiz: Any
    init_iter: Any
    kernel: Any
    lambda_gnmf: Any
    maxIter: Any
    max_iter_snmf: Any
    method_init: Any
    min_corr: Any
    min_pnr: Any
    nIter: Any
    nb: Any
    normalize_init: Any
    options_local_NMF: Any
    perc_baseline_snmf: Any
    ring_size_factor: Any
    rolling_length: Any
    rolling_sum: Any
    seed_method: Any
    sigma_smooth_snmf: Any
    ssub: Any
    ssub_B: Any
    tsub: Any


class SpatialParameters(BaseModel):
    dist: Any
    expandCore: Any
    extract_cc: Any
    maxthr: Any
    medw: Any
    method_exp: Any
    method_ls: Any
    n_pixels_per_process: Any
    nb: Any
    normalize_yyt_one: Any
    nrgthr: Any
    num_blocks_per_run_spat: Any
    se: Any
    ss: Any
    thr_method: Any
    update_background_components: Any

# noinspection DuplicatedCode
class TemporalParameters(BaseModel):
    ITER: Any
    bas_nonneg: Any
    block_size_temp: Any
    fudge_factor: Any
    lags: Any
    optimize_g: Any
    method_deconvolution: Any
    nb: Any
    noise_method: Any
    noise_range: Any
    num_blocks_per_run_temp: Any
    p: Any
    s_min: Any
    solvers: Any
    verbosity: Any


class MergingParameters(BaseModel):
    do_merge: Any
    merge_thr: Any
    merge_parallel: Any


class QualityParameters(BaseModel):
    SNR_lowest: Any
    cnn_lowest: Any
    gSig_range: Any
    min_SNR: Any
    min_cnn_thr: Any
    rval_lowest: Any
    rval_thr: Any
    use_cnn: Any
    use_ecc: Any
    max_ecc: Any


class OnlineParameters(BaseModel):
    N_samples_exceptionality: Any
    batch_update_suff_stat: Any
    dist_shape_update: Any
    ds_factor: Any
    epochs: Any
    expected_comps: Any
    full_XXt: Any
    init_batch: Any
    init_method: Any
    iters_shape: Any
    max_comp_update_shape: Any
    max_num_added: Any
    max_shifts_online: Any
    min_SNR: Any
    min_num_trial: Any
    minibatch_shape: Any
    minibatch_suff_stat: Any
    motion_correct: Any
    movie_name_online: Any
    normalize: Any
    n_refit: Any
    num_times_comp_updated: Any
    opencv_codec: Any
    path_to_model: Any
    ring_CNN: Any
    rval_thr: Any
    save_online_movie: Any
    show_movie: Any
    simultaneously: Any
    sniper_mode: Any
    stop_detection: Any
    test_both: Any
    thresh_CNN_noisy: Any
    thresh_fitness_delta: Any
    thresh_fitness_raw: Any
    thresh_overlap: Any
    update_freq: Any
    update_num_comps: Any
    use_corr_img: Any
    use_dense: Any
    use_peak_max: Any
    W_update_factor: Any


class MotionParameters(BaseModel):
    border_nan: Any
    gSig_filt: Any
    is3D: Any
    max_deviation_rigid: Any
    max_shifts: Any
    min_mov: Any
    niter_rig: Any
    nonneg_movie: Any
    num_frames_split: Any
    num_splits_to_process_els: Any
    num_splits_to_process_rig: Any
    overlaps: Any
    pw_rigid: Any
    shifts_opencv: Any
    splits_els: Any
    splits_rig: Any
    strides: Any
    upsample_factor_grid: Any
    use_cuda: Any
    indices: Any


class RingCNNParameters(BaseModel):
    n_channels: Any
    use_bias: Any
    use_add: False
    pct: Any
    patience: Any
    max_epochs: Any
    width: Any
    loss_fn: Any
    lr: Any
    lr_scheduler: Any
    path_to_model: Any
    remove_activity: Any
    reuse_model: Any


class CNMFParameters(BaseModel):
    ...
