from pathlib import Path
import numpy as np
from exporgo.organization.experiment import Experiment
from tqdm import tqdm
import joblib
from caiman import load
from caiman.mmapping import save_memmap, save_memmap_join
from caiman.source_extraction.cnmf import cnmf


_TEMPORARY_CAIMAN_DIRECTORY = Path(R"C:\Users\Yuste\caiman_data\temp")


def batch_convert_ndarray_to_memmap(files: list[str], delete_original: bool = False) -> list[Path]:
    """
    Converts a list of numpy arrays to memory mapped files
    """
    return [convert_ndarray_to_memmap(Path(file), delete_original) for file in
            tqdm(files, total=len(files), desc="Memory-mapping numpy files")]


def convert_ndarray_to_memmap(file: Path, delete_original: bool = False) -> Path:
    """
    Converts a numpy array to a memory mapped file
    """
    data = np.load(file, allow_pickle=False)
    filename = file.with_stem(file.stem + create_mmap_id(data)).with_suffix(".mmap")
    memory_map_ndarray(filename, data)
    if delete_original:
        file.unlink()
    return filename


def create_mmap_id(data: np.ndarray):
    return f"_d1_{data.shape[1]}_d2_{data.shape[2]}_d3_1_order_F_frames_{data.shape[0]}"


def memory_map_ndarray(file: Path | str, data: np.ndarray) -> None:
    """
    Memory maps a numpy array to a file
    """
    tensor_shape = data.shape
    pixels_per_frame = np.prod(tensor_shape[1:])
    frames = data.shape[0]
    squeezed_shape = (pixels_per_frame, frames)
    # noinspection PyTypeChecker
    mapped_data = np.memmap(filename=file, dtype=np.float32, mode="w+", shape=squeezed_shape, order="F")
    for idx, page in enumerate(data):
        mapped_data[:, idx] = np.reshape(page,
                                         (pixels_per_frame, ),
                                         order="F").astype(np.float32)
    mapped_data.flush()


def calculate_gsig(expected_half_size: int, microns_per_pixel: float) -> tuple[int, int]:
    """
    Calculates the gsig value for the motion correction algorithm
    """
    expected_half_size_in_pixels = convert_microns_to_pixels(expected_half_size, microns_per_pixel)
    return expected_half_size_in_pixels, expected_half_size_in_pixels


def calculate_gsiz(expected_half_size: int, microns_per_pixel: float) -> tuple[int, int]:
    """
    Calculates the gsiz value for the motion correction algorithm
    """
    expected_half_size_in_pixels = convert_microns_to_pixels(expected_half_size, microns_per_pixel)
    gsiz = 2 * expected_half_size_in_pixels + 1
    return gsiz, gsiz


def calculate_max_shift(max_shift_percentage: float, patch_size_in_microns: float, microns_per_pixel: float) \
        -> np.ndarray:
    """
    Calculates the maximum shift in pixels given a percentage of the patch size
    """
    patch_size_in_pixels = convert_microns_to_pixels(patch_size_in_microns, microns_per_pixel)
    max_shift = int(round(max_shift_percentage * patch_size_in_pixels))
    return np.array([max_shift, max_shift], dtype=np.int8)


def calculate_motion_patches(patch_size_in_microns: float, microns_per_pixels: float, overlap_percentage: float) \
        -> tuple[tuple[int, int], tuple[int, int]]:
    """
    Calculates the number of patches in x and y given a patch size in microns and overlap percentage
    """
    patch_size_in_pixels = convert_microns_to_pixels(patch_size_in_microns, microns_per_pixels)
    strides = (round((1-overlap_percentage) * patch_size_in_pixels),
               round((1-overlap_percentage) * patch_size_in_pixels))
    overlaps = (round(overlap_percentage * patch_size_in_pixels),
                round(overlap_percentage * patch_size_in_pixels))
    return strides, overlaps


def calculate_cnmf_patches(patch_size_in_microns: float, microns_per_pixels: float, expected_neuron_diameter: float) \
        -> tuple[tuple[int, int], tuple[int, int]]:
    """
    Calculates the number of patches in x and y given a patch size in microns and overlap percentage
    """
    patch_size_in_pixels = convert_microns_to_pixels(patch_size_in_microns, microns_per_pixels)
    rf = int(round(patch_size_in_pixels // 2))
    stride = int(round(convert_microns_to_pixels(expected_neuron_diameter, microns_per_pixels)))
    return (rf, rf), (stride, stride)


def convert_pixels_microns(pixels: int, microns_per_pixel: float) -> float:
    """
    Converts pixels to microns
    """
    return pixels * microns_per_pixel


def convert_microns_to_pixels(microns: float, microns_per_pixel: float) -> int:
    """
    Converts microns to pixels
    """
    return round(microns / microns_per_pixel)


def get_temporary_files() -> list[str]:
    """
    Gets all the temporary files in the caiman data directory
    """
    return [str(file) for file in _TEMPORARY_CAIMAN_DIRECTORY.glob("*.mmap")]


def get_compiled_file(base_name: str = "memmap_") -> str:
    """
    Gets the compiled file in the caiman data directory
    """
    temporary_files = get_temporary_files()

    for file in temporary_files:
        if base_name in file:
            return file


def get_imaging_files(experiment: Experiment) -> list[str]:
    imaging_files = [str(file) for file in experiment.find("*images0*.npy")]
    return imaging_files


def get_mapped_files(experiment: Experiment) -> list[str]:
    mapped_files = [str(file) for file in experiment.find("*images0*.mmap")]
    return mapped_files


def produce_corrected_compiled_file(results: cnmf,
                                    imaging_files: list[str],
                                    delete_originals: bool = False) -> str:
    shifts = results.estimates.shifts
    base_name = Path(imaging_files[0]).stem.split("_")[0]
    start_idx = 0
    end_idx = 0
    new_files = []
    for file in tqdm(imaging_files, total=len(imaging_files), desc="Saving corrected memmap files", colour="yellow"):
        frames = int(file.split("frames_")[-1].split(".")[0])
        end_idx += frames
        new_files.append(save_memmap([file], base_name=base_name, xy_shifts=shifts[start_idx:end_idx], order="C"))
        start_idx = end_idx
        if delete_originals:
            Path(file).unlink()

    compiled_name = save_memmap_join(new_files, base_name="compiled_" + base_name)
    return compiled_name

