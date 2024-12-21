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
    fnames : list of str
        list of complete paths to files that need to be processed

    dims : tuple of int
        dimension of the FOV in pixels

    fr : float
        imaging frame rate in frames per second

    decay_time : float
        length of a typical transient in seconds

    dxy : tuple of float
        spatial resolution of FOV in pixels per um

    var_name_hdf5 : str
        if loading from hdf5 name of the variable to load

    caiman_version : str
        version of CaImAn being used. Please do not override this

    last_commit : str
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


class PatchParameters(BaseModel):
    ...


class PreprocessParameters(BaseModel):
    ...


class InitParameters(BaseModel):
    ...


class SpatialParameters(BaseModel):
    ...


class TemporalParameters(BaseModel):
    ...


class MergingParameters(BaseModel):
    ...


class QualityParameters(BaseModel):
    ...


class OnlineParameters(BaseModel):
    ...


class MotionParameters(BaseModel):
    ...


class RingCNNParameters(BaseModel):
    ...


class CNMFParameters(BaseModel):
    ...
