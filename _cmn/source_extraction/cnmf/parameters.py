from pydantic import BaseModel, Field, field_validator, model_validator, ConfigDict, ValidationError
from typing import Any, Literal, Callable
from os import path
from dataclasses import dataclass
from caiman.base.movies import get_file_size
from caiman.utils.utils import get_caiman_version
# from caiman.paths import caiman_datadir
from pkg_resources import get_distribution
from types import MappingProxyType
import numpy as np
from functools import wraps
from psutil import cpu_count

#def get_distribution(*args, **kwargs) -> object:
#    class MOCKCAIMAN:
#        version = "1.0.0"
#    return MOCKCAIMAN()


@dataclass(frozen=True, init=False)
class CONSTANTS:
    """
    Namespace for constants used in the parameter model.
    
    Attributes
    ----------
    SUPPORTED_FILE_EXTENSIONS: tuple[str]
        caiman's supported file extensions

    MODEL_CONFIG: MappingProxyType
        pydantic model configuration (read-only, must copy to modify)
    """
    #: caiman's supported file extensions
    SUPPORTED_FILE_EXTENSIONS: tuple[str, ...] = (".avi", ".h5", ".hdf5", ".mmap", ".n5", ".npy", ".tif", ".tiff", ".zarr")
    
    #: pydantic model configuration
    MODEL_CONFIG: MappingProxyType = MappingProxyType({
        # allow arbitrary types to be passed to the model (i.e., non-stdlib)
        "arbitrary_types_allowed": True,
        # defer model building until the first attribute is accessed (i.e., lazy loading)
        "defer_build": True,
        # ignore extra fields passed as arguments to the model
        "extra": "ignore",
        # do not validate fields when they are assigned, set to True in the post-initialization method
        "validate_assignment": False, 
    })


def _consistency_validator(field: str) -> Callable:
    """
    Decorator to tag a method as a consistency validator. This decorator is used to tag methods that validate the
    consistency of the fields in the model with respect to other models. This tag indicates a collection of validators
    that ought to be called by the global model validator :class:`CaimanParameters` after all other validators have
    been called. All consistency validators must take :class:`DataParameters` as an argument.

    Parameters
    ----------
    field: str
        name of the field that the method validates

    Returns
    -------
    Callable
    """
    @wraps
    def tagged_method(func: Callable) -> Callable:
        func.consistency_validator = field
        return func
    return tagged_method


def _get_hash_latest_commit() -> str:
    """
    Get the hash of the latest commit in the repository. Serves as default factory for last_commit field in
    DataParameters

    Returns
    -------
    str
        hash of the latest commit
    """
    return "-".join(get_caiman_version())


def _get_caiman_version() -> str:
    """
    Get the version of CaImAn being used. Serves as default factory for caiman_version field in DataParameters

    Returns
    -------
    str
        version of CaImAn being used
    """
    return get_distribution('caiman').version


def _get_movie_shapes(fnames: tuple[str], var_name_hdf5: str = None) -> list[tuple]:
    """
    Get the shape of the movies in the list of files.

    Parameters
    ----------
    fnames: list of str
        list of complete paths to files that need to be processed

    Returns
    -------
    list of tuple of int
        movie shapes (*dims, frames) for each file in fnames
    """
    shapes = [get_file_size(fname, var_name_hdf5) for fname in fnames]
    return [(*shape[:-1], shape[-1]) for shape in shapes]


class _CaimanParameterModel(BaseModel):
    """
    Base class for all parameter models in this module. This class sets the model configuration for all parameter
    models to be used in the CaImAn package and adds a post-initialization method to validate all fields in the model
    when their values are changed after initialization. Then configuration is changed post-initialization to alleviate
    repetitive validation checks during initialization.

    Attributes
    ----------
    model_config: ConfigDict
        configuration for the model
    """
    #: model configuration
    model_config = ConfigDict(**CONSTANTS.MODEL_CONFIG)

    def model_post_init(self, *args, **kwargs) -> None:
        """
        Post-initialization function for the model. This function sets the validate_assignment parameter to True
        in the model configuration. This is done to ensure that all fields are validated is the user changes them
        after initialization. By doing so, we can guarantee that no invalid values will be set in the model without
        requiring dedicated setter functions.
        """
        self.model_config["validate_assignment"] = True


class DataParameters(_CaimanParameterModel):
    """
    General params describing the dataset

    Attributes
    ----------
    fnames: tuple of str
        tuple of complete paths to files that need to be processed

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
    """

    #: list of complete paths to files that need to be processed
    fnames: tuple[str] | None = None

    #: dimension of the FOV in pixels
    dims: tuple[int, int] | None = None

    #: number of total frames
    frames: int | None = None

    #: imaging frame rate in frames per second.
    fr: float = Field(default=30.0, gt=0.0)

    #: length of a typical transient in seconds.
    decay_time: float = Field(default=1.0, gt=0.0)

    #: spatial resolution of FOV in pixels per um
    dxy: tuple[float, float] = (1.0, 1.0)

    #: if loading from hdf5 name of the variable to load
    var_name_hdf5: str | None = None

    #: version of CaImAn being used. Please do not override this
    caiman_version: str = Field(default_factory=_get_caiman_version, frozen=True)

    #: hash of last commit in the caiman repo. Please do not override this.
    last_commit: str = Field(default_factory=_get_hash_latest_commit, frozen=True)

    # whether to short circuit the validation process for the movies (they can be expensive to validate)
    _short_circuit = hash("caiman")

    @field_validator("fnames", mode="before")
    @classmethod
    def validate_fnames(cls, fnames: Any) -> list[str] | None:
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

        Raises
        ------
        ValidationError
            if fnames is not a list of strings whose length is greater than 0
        Validation Error
            if any member of fnames does not exist
        ValidationError
            if there is more than one file extension in fnames
        ValidationError
            if any file extensions are not a supported format.
        """

        if fnames is not None:
            # ensure fnames is a list of strings
            if isinstance(fnames, list | tuple | set):
                fnames = tuple([str(fname) for fname in fnames])
            else:
                # turn a single value into list of strings if this is not an appropriate value or type it will be
                # caught by the next assertion
                fnames = [str(fnames), ]

            # ensure all members exist
            assert all([path.exists(fname) for fname in fnames]), "All members of fnames must exist"

            # ensure all members have a supported file extension
            assert(all([fname.endswith(CONSTANTS.SUPPORTED_FILE_EXTENSIONS) for fname in fnames]))

            # check that all movies have the same file extension
            assert len(set([fname.split(".")[-1] for fname in fnames])) == 1, \
                "All movies must have the same file extension"

        return fnames

    @field_validator("dims", mode="before")
    @classmethod
    def validate_nonzero_dimensions(cls, dims: tuple[int, ...] | None) -> tuple[int, ...] | None:
        """
        Validates dimensions are greater than 0 if the dimensions field is not None.

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
        ValidationError
            if any value in dims is less than or equal to 0 if dims is not None
        """
        if dims is not None:
            assert all([value > 0 for value in dims]), "All values in dims must be greater than 0"
        return dims

    @field_validator("frames", mode="before")
    @classmethod
    def validate_nonzero_frames(cls, frames: int | None) -> int |None:
        """
        Validates frames is greater than 0 if the frames field is not None.

        Parameters
        ----------
        frames: int
            number of total frames

        Returns
        -------
        int
            number of total frames

        Raises
        ------
        ValidationError
            if frames is less than or equal to 0 if frames is not None
        """
        if frames is not None:
            assert frames > 0, "frames must be greater than 0"
        return frames

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
        ValidationError
            if any value in dxy is less than or equal to 0
        """
        assert all([value > 0.0 for value in dxy]), "All values in dxy must be greater than 0"
        return dxy

    @model_validator(mode="after")
    def validate_movies(self) -> "DataParameters":
        """
        This function asserts that all movies have consistent dimensions. If multiple filenames are
        provided in `fnames`, then all files EXCEPT the final file must contain the same number of frames.

        Raises
        ------
        ValidationError
            if the dimensions of the FOV are not consistent across all movies
        ValidationError
            if the number of frames in all movies are not consistent
        """

        if self.fnames is not None and hash(self.fnames) != self._short_circuit:
            self._short_circuit = hash(self.fnames)

            # get the shapes of all movies
            movie_shapes = _get_movie_shapes(self.fnames, self.var_name_hdf5)

            # if dimensions are not provided, populate dims field with the dimensions of the first movie
            self.dims = self.dims if self.dims is not None else movie_shapes[0][0]
            # if frames are not provided, populate frames field with the number of frames in the first movie
            self.frames = self.frames if self.frames is not None else sum([shape[-1] for shape in movie_shapes])
            print(f"{self.frames=}")
            print(f"{self.dims=}")
            print(f"{movie_shapes=}")
            # check that all movies have the same dimensions.
            assert all([dims == self.dims for dims, _ in movie_shapes]), \
                (f"All movies must have the same dimensions:\n"
                 f"{zip(self.fnames, [dims for dims in movie_shapes])}")

            # check that all movies have the same number of frames
            assert len(set([shape for _, shape in movie_shapes])) == 1, \
                (f"All movies EXCEPT for the final movie must have the same number of frames:\n"
                 f"{zip(self.fnames, [shape[-1] for shape in movie_shapes])}")

        return self


# noinspection DuplicatedCode
class PatchParameters(BaseModel):
    """
    Parameters for patch processing.

    Attributes
    ----------
    border_pix: int, default: 0
        Number of pixels to exclude around each border.

    del_duplicates: bool, default: False
        Delete duplicate components in the overlapping regions between neighboring patches. If False,
        then merging is used.

    in_memory: bool, default: True
        Whether to load patches in memory

    low_rank_background: bool, default: True
        Whether to update the background using a low rank approximation.
        If False all the nonzero elements of the background components are updated using hals
        (to be used with one background per patch)

    memory_fact: float, default: 1
        unitless number for increasing the amount of available memory

    n_processes: int
        Number of processes used for processing patches in parallel

    nb_patch: int, default: 1
        Number of (local) background components per patch

    only_init: bool, default: True
        whether to run only the initialization

    p_patch: int, default: 0
        order of AR dynamics when processing within a patch

    remove_very_bad_comps: bool, default: True
        Whether to remove (very) bad quality components during patch processing

    rf: int or list or None, default: None
        Half-size of patch in pixels. If None, no patches are constructed and the whole FOV is processed jointly.
        If list, it should be a list of two elements corresponding to the height and width of patches

    skip_refinement: bool, default: False
        Whether to skip refinement of components

    p_ssub: float, default: 2
        Spatial downsampling factor

    stride: int or None, default: None
        Overlap between neighboring patches in pixels.

    p_tsub: float, default: 2
        Temporal downsampling factor
    """

    #: model configuration
    model_config = ConfigDict(**CONSTANTS.MODEL_CONFIG)

    #: Number of pixels to exclude around each border.
    border_pix: int = Field(0, ge=0)

    #: Delete duplicate components in the overlapping regions between neighboring patches.
    del_duplicates: bool = False

    #: Whether to load patches in memory
    in_memory: bool = True

    #: Whether to update the background using a low rank approximation.
    low_rank_background: bool = True

    #: unitless number for increasing the amount of available memory
    memory_fact: float = Field(1.0, ge=1.0)

    #: Number of processes used for processing patches in parallel
    n_processes: int = Field(1, ge=1.0)

    #: Number of (local) background components per patch
    nb_patch: int = Field(1, ge=0)

    #: whether to run only the initialization
    only_init: bool = True

    #: order of AR dynamics when processing within a patch
    p_patch: Literal[0, 1, 2] = 0

    #: Whether to remove (very) bad quality components during patch processing
    remove_very_bad_comps: bool = True

    #: Half-size of patch in pixels. If None, no patches are constructed and the whole FOV is processed jointly.
    rf: int | tuple[int, int] | None = None

    #: Whether to skip refinement of components
    skip_refinement: bool = False

    #: Spatial downsampling factor
    p_ssub: int = Field(2, ge=1)

    # Overlap between neighboring patches in pixels.
    stride: int | None

    # Temporal downsampling factor
    p_tsub: int = Field(2, ge=1)

    @field_validator("n_processes", mode="after")
    @classmethod
    def validate_n_processes(cls, n_processes: int) -> int:
        """
        Validate n_processes is less than or equal to the number of available CPUs.

        Parameters
        ----------
        n_processes: int
            Number of processes used for processing patches in parallel

        Returns
        -------
        int

        Raises
        ------
        ValidationError
            if n_processes is greater than the number of available CPUs
        """
        assert n_processes <= cpu_count(), \
            (f"n_processes must be less than or equal to the number of available CPUs\n"
             f"{n_processes=}, {cpu_count()=}")
        return n_processes

    @_consistency_validator("border_pix")
    def validate_border_pix(self, data_parameters: DataParameters) -> None:
        """
        Validate border_pix is less than 1/2th smallest dimension of the FOV.

        Parameters
        ----------
        data_parameters: DataParameters
            data parameters to validate with respect to

        Raises
        ------
        ValidationError
            if border_pix is greater than or equal to the minimum dimension of the FOV
        """
        try:
            assert self.border_pix < min(data_parameters.dims) / 2, \
                "border_pix must be less than the minimum dimension of the FOV"
        except AssertionError as exc:
            raise ValidationError(exc)

    @_consistency_validator("rf")
    def validate_rf(self, data_parameters: DataParameters) -> None:
        """
        Validate rf is less than or equal to 1/2th the minimum dimension of the FOV and
        that there is exactly one rf value for each dimension of the FOV.

        Parameters
        ----------
        data_parameters: DataParameters
            data parameters to validate with respect to

        Raises
        ------
        ValidationError
            if rf is greater than or equal to the 1/2th the minimum dimension of the FOV
        """
        try:
            if isinstance(self.rf, int):
                self.rf = (self.rf, self.rf)
            assert self.rf < min(data_parameters.dims) / 2, \
                "rf must be less than the minimum dimension of the FOV"
        except AssertionError as exc:
            raise ValidationError(exc)

    @model_validator(mode="after")
    def validate_rf_stride_consistent(self) -> "PatchParameters":
        """
        Validate that the stride is less than or equal to the rf.

        Raises
        ------
        ValidationError
            if stride is greater than or equal to rf
        """
        if self.rf is not None and self.stride is not None:
            assert all([self.stride < rf for rf in self.rf]), "stride must be less than rf"
        return self


class PreprocessParameters(BaseModel):
    """
    Parameters for preprocessing the data.

    Attributes
    ----------
    check_nan: bool, default: True
        whether to check for NaNs

    compute_g: bool, default: False
        whether to estimate global time constant

    include_noise: bool, default: False
            flag for using noise values when estimating g

    lags: int, default: 5
        number of lags to be considered for time constant estimation

    max_num_samples_fft: int, default: 3*1024
        Chunk size for computing the PSD of the data (for memory considerations)

    n_pixels_per_process: int, default: 1000
        Number of pixels to be allocated to each process

    noise_method: 'mean'|'median'|'logmexp', default: 'mean'
        PSD averaging method for computing the noise std

    noise_range: (float, float), default: [.25, .5]
        range of normalized frequencies over which to compute the PSD for noise determination

    p: int, default: 2
         order of AR indicator dynamics

    pixels: list, default: None
         pixels to be excluded due to saturation

    sn: np.ndarray or None, default: None
        noise level for each pixel
    """

    #: model configuration
    model_config = ConfigDict(**CONSTANTS.MODEL_CONFIG)

    #: whether to check for NaNs
    check_nan: bool = True

    #: whether to estimate global time constant
    compute_g: bool = False

    #: flag for using noise values when estimating g
    include_noise: bool = False

    #: number of lags to be considered for time constant estimation
    lags: int = Field(5, gt=0)

    #: Chunk size for computing the PSD of the data (for memory considerations)
    max_num_samples_fft: int = Field(3 * 1024, gt=0)

    #: Number of pixels to be allocated to each process
    n_pixels_per_process: int = 1000

    #: Noise averaging method for computing the noise std
    noise_method: Literal["mean", "median", "logmexp"] = "mean"

    #: range of normalized frequencies over which to compute the PSD for noise determination
    noise_range: tuple[float, float] = (.25, .5)

    #: order of AR indicator dynamics
    p: int = Field(2, ge=0, le=2)

    #: pixels to be excluded due to saturation
    pixels: list[int] | None = None

    #: noise level for each pixel
    sn: np.ndarray | None = None

    @field_validator("sn", mode="before")
    @classmethod
    def validate_sn(cls, sn: np.ndarray | None) -> np.ndarray | None:
        if sn is not None:
            sn = np.asarray(sn, dtype=np.float32)
            assert sn.ndim == 1, \
                ("sn must be a 1D ndarray\n"
                 "{sn.ndim=}")
        return sn

    @field_validator("max_num_samples_fft", mode="after")
    @classmethod
    def validate_max_num_samples_fft(cls, max_num_samples_fft: int) -> int:
        assert max_num_samples_fft % 1024 == 0, \
            (f"max_num_samples_fft must be a multiple of 1024\n"
             f"{max_num_samples_fft=}")
        return max_num_samples_fft

    @field_validator("noise_range", mode="after")
    @classmethod
    def validate_noise_range(cls, noise_range: tuple[float, float]) -> tuple[float, float]:
        assert all([0.0 <= value <= 1.0 for value in noise_range]), \
            (f"noise_range values must be between 0 and 1\n"
             f"{noise_range=}")
        return noise_range


class InitParameters(BaseModel):
    """
    Parameters for initialization.

    Attributes
    ----------
    K: int, default: 30
        number of components to be found (per patch or whole FOV depending on whether rf=None)

    SC_kernel: {'heat', 'cos', 'binary'}, default: 'heat'
        kernel for graph affinity matrix

    SC_sigma: float, default: 1
        variance for SC kernel

    SC_thr: float, default: 0,
        threshold for affinity matrix

    SC_normalize: bool, default: True
        standardize entries prior to computing the affinity matrix

    SC_use_NN: bool, default: False
        sparsify affinity matrix by using only nearest neighbors

    SC_nnn: int, default: 20
        number of nearest neighbors to use

    alpha_snmf: float, default: 0.5
        sparse NMF sparsity regularization weight

    center_psf: bool, default: False
        whether to use 1p data processing mode. Set to true for 1p

    gSig: [int, int], default: [5, 5]
        radius of average neurons (in pixels)

    gSiz: [int, int], default: [int(round((x * 2) + 1)) for x in gSig],
        half-size of bounding box for each neuron

    init_iter: int, default: 2
        number of iterations during corr_pnr (1p) initialization

    kernel: np.array or None, default: None
        user specified template for greedyROI

    lambda_gnmf: float, default: 1.
        regularization weight for graph NMF

    maxIter: int, default: 5
        number of HALS iterations during initialization

    max_iter_snmf : int, default: 500
        maximum number of iterations for sparse NMF initialization

    method_init: 'greedy_roi'|'corr_pnr'|'sparse_NMF'|'local_NMF' default: 'greedy_roi'
        initialization method. use 'corr_pnr' for 1p processing and 'sparse_NMF' for dendritic processing.

    min_corr: float, default: 0.85
        minimum value of correlation image for determining a candidate component during corr_pnr

    min_pnr: float, default: 20
        minimum value of psnr image for determining a candidate component during corr_pnr

    nIter: int, default: 5
        number of rank-1 refinement iterations during greedy_roi initialization

    nb: int, default: 1
        number of background components

    normalize_init: bool, default: True
        whether to equalize the movies during initialization

    options_local_NMF: dict
        dictionary with parameters to pass to local_NMF initializer

    perc_baseline_snmf: float, default: 20
        percentile to be removed from the data in sparse_NMF prior to decomposition

    ring_size_factor: float, default: 1.5
        radius of ring (*gSig) for computing background during corr_pnr

    rolling_length: int, default: 100
        width of rolling window for rolling sum option

    rolling_sum: bool, default: True
        use rolling sum (as opposed to full sum) for determining candidate centroids during greedy_roi

    seed_method: str {'auto', 'manual', 'semi'}
        methods for choosing seed pixels during greedy_roi or corr_pnr initialization
        'semi' detects nr components automatically and allows to add more manually
        if running as notebook 'semi' and 'manual' require a backend that does not
        inline figures, e.g. %matplotlib tk

    sigma_smooth_snmf : (float, float, float), default: (.5,.5,.5)
        std of Gaussian kernel for smoothing data in sparse_NMF

    ssub: float, default: 2
        spatial downsampling factor

    ssub_B: float, default: 2
        downsampling factor for background during corr_pnr

    tsub: float, default: 2
        temporal downsampling factor
    """
    #: model configuration
    model_config = ConfigDict(**CONSTANTS.MODEL_CONFIG)

    #: number of components to be found
    K: int = Field(0, ge=0)

    #: kernel for graph affinity matrix
    SC_kernel: Literal["heat", "cos", "binary"] = "heat"

    #: variance for SC kernel
    SC_sigma: float = Field(1.0, gt=0.0)

    #: threshold for affinity matrix
    SC_thr: float = Field(0.0, ge=0.0)

    #: standardize entries prior to computing the affinity matrix
    SC_normalize: bool = True

    #: sparsify affinity matrix by using only nearest neighbors
    SC_use_NN: bool = False

    #: number of nearest neighbors to use
    SC_nnn: int = Field(20, gt=0)

    #: sparse NMF sparsity regularization weight
    alpha_snmf: float = Field(0.5, ge=0.0)

    #: whether to use 1p data processing mode. Set to true for 1p
    center_psf: bool = False

    #: radius of average neurons (in pixels)
    gSig: tuple[int, ...] = (5, 5)

    #: half-size of bounding box for each neuron
    gSiz: tuple[int, ...] | None = None

    #: number of iterations during corr_pnr (1p) initialization
    init_iter: int = Field(2, gt=0)

    #: user specified template for greedyROI
    kernel: np.ndarray | None = None

    #: regularization weight for graph NMF
    lambda_gnmf: float = Field(1.0, ge=0.0)

    #: number of HALS iterations during initialization
    maxIter: int = Field(5, gt=0)

    #: maximum number of iterations for sparse NMF initialization
    max_iter_snmf: int = Field(500, gt=0)

    #: initialization method. use 'corr_pnr' for 1p processing and 'sparse_NMF' for dendritic processing.
    method_init: Literal["greedy_roi", "corr_pnr", "sparse_NMF", "local_NMF"] = "greedy_roi"

    #: minimum value of correlation image for determining a candidate component during corr_pnr
    min_corr: float = Field(0.85, ge=0.0, le=1.0)

    #: minimum value of psnr image for determining a candidate component during corr_pnr
    min_pnr: float = Field(20, ge=0)

    #: number of rank-1 refinement iterations during greedy_roi initialization
    nIter: int = Field(1, gt=0)

    #: number of background components
    nb: int = Field(1, ge=0)

    #: whether to equalize the movies during initialization
    normalize_init: bool = True

    #: dictionary with parameters to pass to local_NMF initializer
    options_local_NMF: dict = Field(default_factory=dict)

    #: percentile to be removed from the data in sparse_NMF prior to decomposition
    perc_baseline_snmf: float = Field(20, ge=0.0, le=100.0)

    #: radius of ring (*gSig) for computing background during corr_pnr
    ring_size_factor: float = Field(1.5, gt=0.0)

    #: width of rolling window for rolling sum option
    rolling_length: int = Field(100, gt=0)

    #: use rolling sum (as opposed to full sum) for determining candidate centroids during greedy_roi
    rolling_sum: bool = True

    #: methods for choosing seed pixels during greedy_roi or corr_pnr initialization
    seed_method: Literal["auto", "manual", "semi"] = "auto"

    #: std of Gaussian kernel for smoothing data in sparse_NMF
    sigma_smooth_snmf: tuple[float, float, float] = (.5, .5, .5)

    #: spatial downsampling factor
    ssub: float = Field(2, ge=1)

    #: downsampling factor for background during corr_pnr
    ssub_B: float = Field(2, ge=1)

    #: temporal downsampling factor
    tsub: float = Field(2, ge=1)


class SpatialParameters(BaseModel):
    """
    Parameters for spatial processing.

    Attributes
    ----------
    dist: float, default: 3
        expansion factor of ellipse

    expandCore: morphological element, default: None(?)
        morphological element for expanding footprints under dilate

    extract_cc: bool, default: True
        whether to extract connected components during thresholding
        (might want to turn to False for dendritic imaging)

    maxthr: float, default: 0.1
        Max threshold

    medw: (int, int) default: None
        window of median filter (set to (3,)*len(dims) in cnmf.fit)

    method_exp: 'dilate'|'ellipse', default: 'dilate'
        method for expanding footprint of spatial components

    method_ls: 'lasso_lars'|'nnls_L0', default: 'lasso_lars'
        'nnls_L0'. Nonnegative least square with L0 penalty
        'lasso_lars' lasso lars function from scikit learn

    n_pixels_per_process: int, default: 1000
        number of pixels to be processed by each worker

    nb: int, default: 1
        number of global background components. Do not set this directly; modify it in init.

    normalize_yyt_one: bool, default: True
        Whether to normalize the C and A matrices so that diag(C*C.T) = 1 during update spatial

    nrgthr: float, default: 0.9999
        Energy threshold

    num_blocks_per_run_spat: int, default: 20
        Parallelization of A'*Y operation

    se: np.array or None, default: None
         Morphological closing structuring element (set to np.ones((3,)*len(dims), dtype=np.uint8) in cnmf.fit)

    ss: np.array or None, default: None
        Binary element for determining connectivity (set to np.ones((3,)*len(dims), dtype=np.uint8) in cnmf.fit)

    thr_method: 'nrg'|'max', default: 'nrg'
        thresholding method

    update_background_components: bool, default: True
        whether to update the spatial background components
    """
    #: model configuration
    model_config = ConfigDict(**CONSTANTS.MODEL_CONFIG)

    #: expansion factor of ellipse
    dist: float = Field(3, ge=1.0)

    #: morphological element for expanding footprints under dilate
    expandCore: Any = None

    #: whether to extract connected components during thresholding
    extract_cc: bool = True

    #: Max threshold
    maxthr: float = Field(0.1, ge=0.0)

    #: window of median filter (set to (3,)*len(dims) in cnmf.fit)
    medw: tuple[int, ...] | None = None

    #: method for expanding footprint of spatial components
    method_exp: Literal["dilate", "ellipse"] = "dilate"

    #: method for
    method_ls: Literal["lasso_lars", "nnls_L0"] = "lasso_lars"

    #: number of pixels to be processed by each worker
    n_pixels_per_process: int = 1000

    #: number of global background components. Do not set this directly; modify it in init.
    nb: int = Field(1, ge=0)

    #: Whether to normalize the C and A matrices so that diag(C*C.T) = 1 during update spatial
    normalize_yyt_one: bool = True

    #: Energy threshold
    nrgthr: float = Field(0.9999, ge=0.0, le=1.0)

    #: Parallelization of A'*Y operation
    num_blocks_per_run_spat: int = Field(20, ge=1)

    #: Morphological closing structuring element (set to np.ones((3,)*len(dims), dtype=np.uint8) in cnmf.fit)
    se: np.ndarray | None = None

    #: Binary element for determining connectivity (set to np.ones((3,)*len(dims), dtype=np.uint8) in cnmf.fit)
    ss: np.ndarray | None = None

    #: thresholding method
    thr_method: Literal["nrg", "max"] = "nrg"

    #: whether to update the spatial background components
    update_background_components: bool = True


# noinspection DuplicatedCode
class TemporalParameters(BaseModel):
    """
    Parameters for temporal processing.

    Attributes
    ----------
    ITER: int, default: 2
        block coordinate descent iterations

    bas_nonneg: bool, default: True
        whether to set a non-negative baseline (otherwise b >= min(y))

    block_size_temp : int, default: 5000
        Number of pixels to process at the same time for dot product. Reduce if you face memory problems

    fudge_factor: float (close but smaller than 1) default: .96
        bias correction factor for discrete time constants

    lags: int, default: 5
        number of autocovariance lags to be considered for time constant estimation

    optimize_g: bool, default: False
        flag for optimizing time constants

    method_deconvolution: 'oasis'|'cvxpy'|'oasis', default: 'oasis'
        method for solving the constrained deconvolution problem ('oasis','cvx' or 'cvxpy')
        if method cvxpy, primary and secondary (if problem unfeasible for approx solution)

    nb: int, default: 1
        number of global background components. Do not set this directly; modify it in init.

    noise_method: 'mean'|'median'|'logmexp', default: 'mean'
        PSD averaging method for computing the noise std

    noise_range: [float, float], default: [.25, .5]
        range of normalized frequencies over which to compute the PSD for noise determination

    num_blocks_per_run_temp: int, default: 20
        Parallelization of A'*Y operation

    p: 0|1|2, default: 2
        order of AR indicator dynamics

    s_min: float or None, default: None
        Minimum spike threshold amplitude (computed in the code if used).

    solvers: 'ECOS'|'SCS', default: ['ECOS', 'SCS']
         solvers to be used with cvxpy, can be 'ECOS','SCS' or 'CVXOPT'

    verbosity: bool, default: False
        whether to be verbose
    """

    #: model configuration
    model_config = ConfigDict(**CONSTANTS.MODEL_CONFIG)

    #: block coordinate descent iterations
    ITER: int = Field(2, ge=1)

    #: whether to set a non-negative baseline (otherwise b >= min(y))
    bas_nonneg: bool = True

    #: Number of pixels to process at the same time for dot product. Reduce if you face memory problems
    block_size_temp: int = Field(5000, ge=1)

    #: bias correction factor for discrete time constants
    fudge_factor: float = Field(.96, gt=0.0, lt=1.0)

    #: number of autocovariance lags to be considered for time constant estimation
    lags: int = Field(5, ge=1)

    #: flag for optimizing time constants
    optimize_g: bool = False

    #: method for solving the constrained deconvolution problem ('oasis','cvx' or 'cvxpy')
    method_deconvolution: Literal["oasis", "cvx", "cvxpy"] = "oasis"

    #: number of global background components. Do not set this directly; modify it in init.
    nb: int = Field(1, ge=0)

    #: PSD averaging method for computing the noise std
    noise_method: Literal["mean", "median", "logmexp"] = "mean"

    #: range of normalized frequencies over which to compute the PSD for noise determination
    noise_range: tuple[float, float] = (.25, .5)

    #: Parallelization of A'*Y operation
    num_blocks_per_run_temp: int = Field(20, ge=1)

    #: order of AR indicator dynamics
    p: int = Field(2, ge=0, le=2)

    #: Minimum spike threshold amplitude (computed in the code if used).
    s_min: float | None = None

    #: solvers to be used with cvxpy, can be 'ECOS','SCS' or 'CVXOPT'
    solvers: Literal["ECOS", "SCS", "CVXOPT"] = ["ECOS", "SCS"]

    #: whether to be verbose
    verbosity: bool = False

    @field_validator("noise_range", mode="after")
    @classmethod
    def validate_noise_range(cls, noise_range: tuple[float, float]) -> tuple[float, float]:
        assert all([0.0 <= value <= 1.0 for value in noise_range]), \
            (f"noise_range values must be between 0 and 1\n"
             f"{noise_range=}")
        return noise_range


class MergingParameters(BaseModel):
    """
    Parameters for merging components.

    Attributes
    ----------
    do_merge: bool
        flag for merging components

    merge_thr: float
        threshold for merging components

    merge_parallel: bool
        flag for parallel merging
    """
    #: model configuration
    model_config = ConfigDict(**CONSTANTS.MODEL_CONFIG)

    #: Whether or not to merge components
    do_merge: bool = True

    #: Trace correlation threshold for merging two components.
    merge_thr: float = Field(0.8, ge=0.0, le=1.0)

    #: Perform merging in parallel
    merge_parallel: bool = False


class QualityParameters(BaseModel):
    """
    Quality parameters for component evaluation.

    Attributes
    ----------
    SNR_lowest: float
        minimum required trace SNR

    cnn_lowest: float
        minimum required CNN threshold

    gSig_range: list of int
        gSig scale values for CNN classifier

    min_SNR: float
        trace SNR threshold

    min_cnn_thr: float
        CNN classifier threshold

    rval_lowest: float
        minimum required space correlation

    rval_thr: float
        space correlation threshold.

    use_cnn: bool
        flag for using the CNN classifier
    """

    #: model configuration
    model_config = ConfigDict(**CONSTANTS.MODEL_CONFIG)

    #: minimum required trace SNR
    SNR_lowest: float = Field(default=0.5, ge=0.0)

    #: minimum required CNN threshold
    cnn_lowest: float = Field(default=0.1, ge=0.0, le=1.0)

    #:  gSig scale values for CNN classifier
    gSig_range: list[int] | None

    #: trace SNR threshold.
    min_SNR: float = Field(default=2.0, ge=0.0)

    #: CNN classifier threshold.
    min_cnn_thr: float = Field(default=0.90, ge=0.0, le=1.0)

    #: minimum required space correlation.
    rval_lowest: float = -1.0

    #: space correlation threshold.
    rval_thr: float = 0.8

    #: flag for using the CNN classifier
    use_cnn: bool = True

    #: (undocumented)
    use_ecc: bool = Field(False, frozen=True)

    #: (undocumented)
    max_ecc: int = Field(3, frozen=True)


class OnlineParameters(BaseModel):
    ...


class MotionParameters(BaseModel):
    """
    Parameters for motion correction.

    Attributes
    ----------
    border_nan: bool or str, default: 'copy'
        flag for allowing NaN in the boundaries. True allows NaN, whereas 'copy' copies the value of the
        nearest data point.

    gSig_filt: int or None, default: None
        size of kernel for high pass spatial filtering in 1p data. If None no spatial filtering is performed

    is3D: bool, default: False
        flag for 3D recordings for motion correction

    max_deviation_rigid: int, default: 3
        maximum deviation in pixels between rigid shifts and shifts of individual patches

    max_shifts: (int, int), default: (6,6)
        maximum shifts per dimension in pixels.

    min_mov: float or None, default: None
        minimum value of movie. If None it get computed.

    niter_rig: int, default: 1
        number of iterations rigid motion correction.

    nonneg_movie: bool, default: True
        flag for producing a non-negative movie.

    num_frames_split: int, default: 80
        split movie every x frames for parallel processing

    num_splits_to_process_rig, default: None
        (Undocumented, changing this likely to break the code - FIXME why is this a parameter then?)

    overlaps: (int, int), default: (24, 24)
        overlap between patches in pixels in pw-rigid motion correction.

    pw_rigid: bool, default: False
        flag for performing pw-rigid motion correction.

    shifts_opencv: bool, default: True
        flag for applying shifts using cubic interpolation (otherwise FFT)

    splits_els: int, default: 14
        number of splits across time for pw-rigid registration.

    splits_rig: int, default: 14
        number of splits across time for rigid registration.

    strides: (int, int), default: (96, 96)
        how often to start a new patch in pw-rigid registration. Size of each patch will be strides + overlaps

    upsample_factor_grid" int, default: 4
        motion field upsampling factor during FFT shifts.

    use_cuda: bool, default: False
        flag for using a GPU.

    indices: tuple(slice), default: (slice(None), slice(None))
        Use that to apply motion correction only on a part of the FOV

    """

    #: model configuration
    model_config = ConfigDict(**CONSTANTS.MODEL_CONFIG)

    #: flag for allowing NaN in the boundaries. True allows NaN, whereas 'copy' copies the value of the nearest data point.
    border_nan: Literal[True, False, "copy"] = "copy"

    #: size of kernel for high pass spatial filtering in 1p data. If None no spatial filtering is performed
    gSig_filt: int | None = None

    #: flag for 3D recordings for motion correction
    is3D: bool = False

    #: maximum deviation in pixels between rigid shifts and shifts of individual patches
    max_deviation_rigid: int = Field(3, ge=0)

    #: maximum shifts per dimension in pixels.
    max_shifts: tuple[int, ...] = (6, 6)

    #: minimum value of movie. If None it get computed.
    min_mov: float | None = None

    #: number of iterations rigid motion correction.
    niter_rig: int = Field(1, ge=1)

    #: flag for producing a non-negative movie.
    nonneg_movie: bool = True

    #: split movie every x frames for parallel processing
    num_frames_split: int = Field(80, ge=1)

    #: (Undocumented, changing this likely to break the code - FIXME why is this a parameter then?)
    num_splits_to_process_rig: None = Field(None, frozen=True)

    #: overlap between patches in pixels in pw-rigid motion correction.
    overlaps: tuple[int, int] = (24, 24)

    #: flag for performing pw-rigid motion correction.
    pw_rigid: bool = False

    #: flag for applying shifts using cubic interpolation (otherwise FFT)
    shifts_opencv: bool = True

    #: number of splits across time for pw-rigid registration.
    splits_els: int = Field(14, ge=1)

    #: number of splits across time for rigid registration.
    splits_rig: int = Field(14, ge=1)

    #: how often to start a new patch in pw-rigid registration. Size of each patch will be strides + overlaps
    strides: tuple[int, ...] = (96, 96)

    #: motion field upsampling factor during FFT shifts.
    upsample_factor_grid: int = Field(4, ge=1)

    #: flag for using a GPU.
    use_cuda: bool = False

    #: Use that to apply motion correction only on a part of the FOV
    indices: tuple[slice, ...] = (slice(None), slice(None))


class RingCNNParameters(BaseModel):
    """
    Parameters for RingCNN.

    Attributes
    ----------
    n_channels: int, default: 2
        Number of "ring" kernels

    use_bias: bool, default: False
        Flag for using bias in the convolutions

    use_add: bool, default: False
        Flag for using an additive layer

    pct: float between 0 and 1, default: 0.01
        Quantile used during training with quantile loss function

    patience: int, default: 3
        Number of epochs to wait before early stopping

    max_epochs: int, default: 100
        Maximum number of epochs to be used during training

    width: int, default: 5
        Width of "ring" kernel

    loss_fn: str, default: 'pct'
        Loss function specification ('pct' for quantile loss function,
        'mse' for mean squared error)

    lr: float, default: 1e-3
        (initial) learning rate

    lr_scheduler: function, default: None
        Learning rate scheduler function

    path_to_model: str, default: None
        Path to saved weights (if training then path to saved model weights)

    remove_activity: bool, default: False
        Flag for removing activity of last frame prior to background extraction

    reuse_model: bool, default: False
        Flag for reusing an already trained model (saved in path to model)
    """

    #: model configuration
    model_config = ConfigDict(**CONSTANTS.MODEL_CONFIG)

    #: Number of "ring" kernels
    n_channels: int = Field(2, ge=1)

    #: Flag for using bias in the convolutions
    use_bias: bool = False

    #: Flag for using an additive layer
    use_add: bool = False

    #: Quantile used during training with quantile loss function
    pct: float = Field(0.01, ge=0.0, le=1.0)

    #: Number of epochs to wait before early stopping
    patience: int = Field(3, ge=0)

    #: Maximum number of epochs to be used during training
    max_epochs: int = Field(100, ge=1)

    #: Width of "ring" kernel
    width: int = Field(5, ge=1)

    #: Loss function specification ('pct' for quantile loss function, 'mse' for mean squared error)
    loss_fn: Literal["pct", "mse"] = "pct"

    #: (initial) learning rate
    lr: float = Field(1e-3, gt=0.0)

    #: Learning rate scheduler function
    lr_scheduler: Callable | None = None

    #: Path to saved weights (if training then path to saved model weights)
    path_to_model: str | None = None

    #: Flag for removing activity of last frame prior to background extraction
    remove_activity: bool = False

    #: Flag for reusing an already trained model (saved in path to model)
    reuse_model: bool = False


class CNMFParameters(BaseModel):
    """
    Caiman Parameters

    Attributes
    ----------
    data: DataParameters
        Data parameters

    patch: PatchParameters
        Patch parameters

    preprocessing: PreprocessParameters
        Preprocessing parameters

    init: InitParameters
        Initialization parameters

    spatial: SpatialParameters
        Spatial parameters

    temporal: TemporalParameters
        Temporal parameters

    merging: MergingParameters
        Merging parameters

    quality: QualityParameters
        Quality parameters

    online: OnlineParameters
        Online parameters

    motion: MotionParameters
        Motion parameters

    ring_cnn: RingCNNParameters
        RingCNN parameters

    """
    ...
    #: model configuration
    model_config = ConfigDict(**CONSTANTS.MODEL_CONFIG)

    #: Data parameters
    data: DataParameters = Field(default_factory=DataParameters)

    #: Patch parameters
    patch: PatchParameters = Field(default_factory=PatchParameters)

    #: Preprocessing parameters
    preprocessing: PreprocessParameters = Field(default_factory=PreprocessParameters)

    #: Initialization parameters
    init: InitParameters = Field(default_factory=InitParameters)

    #: Spatial parameters
    spatial: SpatialParameters = Field(default_factory=SpatialParameters)

    #: Temporal parameters
    temporal: TemporalParameters = Field(default_factory=TemporalParameters)

    #: Merging parameters
    merging: MergingParameters = Field(default_factory=MergingParameters)

    #: Quality parameters
    quality: QualityParameters = Field(default_factory=QualityParameters)

    #: Online parameters
    online: OnlineParameters = Field(default_factory=OnlineParameters)

    #: Motion parameters
    motion: MotionParameters = Field(default_factory=MotionParameters)

    #: RingCNN parameters
    ring_cnn: RingCNNParameters = Field(default_factory=RingCNNParameters)
