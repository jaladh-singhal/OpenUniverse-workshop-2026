---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.19.1
kernelspec:
  name: python3
  display_name: python3
  language: python
---

# GW Host

## Learning Goals

By the end of this tutorial, you will be able to :

- Query the OpenUniverse2024 images for a source of interest
- Perform aperture photometry on images
- Generate a light curve
- Display full images and cutouts

## Introduction

The [OpenUniverse2024]((https://arxiv.org/abs/2501.05632)) simulation suite delivers ~70 deg² of matched optical/infrared imagery designed for both the LSST Wide‑Fast‑Deep (WFD) and the Nancy Grace Roman Space Telescope high-latitude survey, enabling joint survey planning and multi-wavelength systematics studies. It incorporates the updated “Diffsky” extragalactic model, extended transient modeling across optical/IR wavelengths, and realistic telescope/instrument effects, producing roughly 400 TB of publicly available synthetic imaging and catalogs. The goal of this project is to enable cross-collaboration and maximize science return from next-generation cosmological surveys by providing a consistent simulated sky observed by multiple observatories.

Tidal Disruption Events (TDEs) occur when a star passes close enough to a supermassive black hole to be torn apart by tidal forces, producing a luminous flare that can outshine the host galaxy for weeks to months.
Identifying and characterizing TDE host galaxies is key to understanding the demographics of supermassive black holes and the galactic environments that produce these rare events.
This notebook demonstrates how to locate a simulated TDE from the OpenUniverse2024 transient input catalog, identify its host galaxy, and extract optical and infrared photometry from Roman and Rubin images to construct a multi-epoch light curve.

### Instructions

This notebook is designed to be run sequentially from top to bottom.  All code is self-contained and relies on publicly accessible data.

### Input

- GW alert? or will the code go out and get one?

### Output

- Light curves of potential host galaxies
- Cutout gallery of potential host galaxies

## Imports

```{code-cell} ipython3
import time
starttime = time.time()
```

```{code-cell} ipython3
# Uncomment the next line to install dependencies if needed.
# !pip install numpy astropy s3fs photutils matplotlib scipy pandas fsspec pyarrow astropy-healpix
```

```{code-cell} ipython3
from astropy.io import fits
import numpy as np
import s3fs
from matplotlib import pyplot as plt
import pandas as pd
from photutils.aperture import CircularAperture, aperture_photometry
from scipy.ndimage import rotate
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.nddata import Cutout2D
from astropy.wcs import WCS
import astropy_healpix as ah
import pyarrow.dataset as ds
import pyarrow.fs
import pyarrow.parquet as pq
import json

import itertools
```

## 1. Explore the OpenUniverse2024 data directories

This section of the tutorial demonstrates how to explore the OpenUniverse2024 data directories directly on S3 and inspect simulated Roman and Rubin images without downloading large datasets locally. It establishes a connection to the public NASA IRSA simulations bucket using s3fs, defines key directory paths for the full Roman and Rubin simulations (not the preview subsets), and illustrates how to browse image files for a selected band and pointing. The accompanying functions — summarize_fits_files() and show_gallery() — provide tools for quickly summarizing FITS file metadata (e.g., number of extensions, pointing information, pixel scale) and for visualizing a small gallery of example images from the chosen directory.

In the prefix you will see that we choose "simple_model" simulations and not "truth" simulations because the simple_model images are the ones with noise and real effects, while "Truth" are noise free, perfect images.

Also in the prefix you will see that we choose the full simulation, not the preview simulation for both Roman and Rubin. Differences between the "full" and "preview" simulations are clarified in the [this](https://arxiv.org/abs/2501.05632) publication

```{code-cell} ipython3
# Setup

# Create a connection to the public NASA IRSA S3 storage bucket using the `s3fs` library.
# By setting `anon=True`, the connection is opened in **anonymous read-only mode**,
# allowing us to list and access public files (such as the OpenUniverse2024 Roman and
# Rubin simulation data) directly from S3 without requiring AWS credentials.

#initialize a general interface to Amazon cloud
s3 = s3fs.S3FileSystem(anon=True)

#general location information
BUCKET_NAME = "nasa-irsa-simulations"
OU_PREFIX = "openuniverse2024"
ROMAN_TDS_PREFIX = "roman/full/RomanTDS/images/simple_model"  #
CATALOG_NAME = "roman_rubin_cats_v1.1.2_faint"

#spcific location information
BAND= "J129"
POINTING = "10190"

#the full path to the data we are interested in exploring
image_directory = f"{BUCKET_NAME}/{OU_PREFIX}/{ROMAN_TDS_PREFIX}/{BAND}/{POINTING}"

#list the contents
s3.ls(image_directory)
```

```{code-cell} ipython3
# open and explore extensions

# how many files are in the bucket?
files = [f"s3://{f}" for f in s3.ls(image_directory)]
print(f"Found {len(files)} files")

#pick one fits file to explore
fname = files[0]

#describe the available extensions in this fits file
with fits.open(fname, use_fsspec=True, fsspec_kwargs={"anon": True}, memmap=False) as hdul:
    print(f"File: {fname}")
    print(f"Number of extensions: {len(hdul)}\n")
    hdul.info()
```

This output lists the structure and contents of one example Roman TDS FITS image from the OpenUniverse2024 dataset. It shows that 18 image files were found in the selected directory, and the examined file contains four extensions: a primary header (no data) followed by three 4088×4088 pixel image planes labeled SCI, ERR, and DQ, which store the science image, per-pixel uncertainty, and data quality mask, respectively. For each extension, the output reports its type, data dimensions, and the first few header keywords to give you a sense of what is in the file.

+++

Let's take a look at a few images to see what we are dealing with.

```{code-cell} ipython3
---
jupyter:
  source_hidden: true
---
def show_gallery(files, max_images=9):
    """
    Display a gallery of FITS images.

    Parameters
    ----------
    files : list of str
        List of S3 URIs to FITS files.
    max_images : int, optional
        Maximum number of images to display (default: 9).
    """
    # Limit the number of images to display
    n_images = min(len(files), max_images)

    # Choose number of columns: up to 3, or equal to n_images if fewer than 4
    ncols = n_images if n_images < 4 else 3

    # Compute number of rows based on total images and columns
    nrows = (n_images + ncols - 1) // ncols

    # Create the subplot grid
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows))
    axes = np.atleast_1d(axes).ravel()  # flatten in case of 1D output

    # Loop through each file and display the image
    for i, f in enumerate(files[:n_images]):
        # Open FITS file directly from S3 (anonymous read)
        with fits.open(f, fsspec_kwargs={"anon": True}, memmap=False) as hdul:
            data = hdul[1].data  # Extract image data
            # Compute robust display scaling (5th–99th percentile)
            vmin, vmax = np.nanpercentile(data, [5, 99])
            # Show image in grayscale with good contrast
            axes[i].imshow(data, origin="lower", cmap="gray", vmin=vmin, vmax=vmax)
            # Title: just the base filename
            axes[i].set_title(f.split("/")[-1], fontsize=8)
            # Remove tick marks and labels
            axes[i].axis("off")

    # Hide any unused subplot axes
    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    # Adjust layout for neat display
    plt.tight_layout()
    plt.show()
```

```{code-cell} ipython3
show_gallery(files)
```

yup, definitely different positions!

+++

## 2. Find a TDE target from the transient catalog

We use the OpenUniverse2024 transient input catalog — the same SNANA parquet files described in the [SED Fitting tutorial](sed_fit) — to find a TDE.
The catalog stores one parquet file per HEALPix region, and TDEs are rare, so not every region will contain one.
We iterate over regions in sorted order and stop at the first TDE we find, using its host galaxy sky position as our search center for the sections that follow.

First, we connect to S3 and list all available SNANA parquet files in the catalog.

```{code-cell} ipython3
fs = pyarrow.fs.S3FileSystem(anonymous=True)
catalog_prefix = f"{BUCKET_NAME}/{OU_PREFIX}/roman/full/{CATALOG_NAME}"

file_info = fs.get_file_info(pyarrow.fs.FileSelector(catalog_prefix, recursive=False))
snana_files = sorted([
    f.path for f in file_info
    if f.base_name.startswith("snana_") and f.base_name.endswith(".parquet")
])

print(f"Found {len(snana_files)} snana parquet files")
```

Next, we scan those files in order, reading each one until we find a row with `model_name == "NON1ASED.TDE-BBFIT"`.
We record the first TDE found and the region it came from, then stop.

```{code-cell} ipython3
tde_row = None
tde_region = None
for path in snana_files:
    df = pq.read_table(path, filesystem=fs).to_pandas()
    mask = df["model_name"] == "NON1ASED.TDE-BBFIT"
    if mask.any():
        tde_row = df[mask].iloc[0]
        # extract region index from filename (e.g. "snana_9921.parquet" → "9921")
        tde_region = path.split("snana_")[1].replace(".parquet", "")
        print(f"Found TDE in region {tde_region}")
        break

if tde_row is None:
    raise RuntimeError("No TDE found in any snana parquet file.")
```

Once we have the TDE, we load the corresponding galaxy info parquet file for that region to look up the host galaxy's sky coordinates, and set the position variables used by Section 3.

```{code-cell} ipython3
galaxy_info_file = f"{catalog_prefix}/galaxy_{tde_region}.parquet"
gal_info = pq.read_table(galaxy_info_file, filesystem=fs).to_pandas()
host_row = gal_info[gal_info["galaxy_id"] == tde_row["host_id"]].iloc[0]

ra_center  = host_row["ra"] * u.deg
dec_center = host_row["dec"] * u.deg
radius_deg = 1 * u.arcsec

print(f"TDE host galaxy: RA={ra_center:.4f}, Dec={dec_center:.4f}")
print(f"Search radius: {radius_deg}")
```

## 3. Data Access
To locate data covering the region identified by the TDE target, we begin by performing a cone search in the existing OpenUniverse2024 Roman + Rubin catalogs. This step identifies all known galaxies within some small radius of the TDE position identified above. The resulting catalog provides positions and IDs for each potential host galaxy.  With those coordinates in hand, we then query the corresponding Roman and Rubin image files that overlap this same region, retrieving only the fits files needed for subsequent photometry and light-curve analysis.

+++

### 3.1 Catalog access
We use the "roman_rubin_cats_v1.1.2_faint" catalog (defined above as CATALOG_NAME) because it provides precise sky positions and unique galaxy IDs for all simulated Roman + Rubin sources, allowing us to later cross-match these galaxies with other derived quantities such as photometry or physical parameters.
We select the full survey rather than the preview version because it covers a larger sky area and represents the more recent, higher-fidelity release of the OpenUniverse2024 simulations.
"faint" in the catalog name refers to the deeper magnitude limit of the simulation.

```{code-cell} ipython3
---
jupyter:
  source_hidden: true
---
def cone_search_catalog(
    ra_center,
    dec_center,
    radius_deg,
    s3_prefix=(
        f"s3://{BUCKET_NAME}/{OU_PREFIX}/roman/full/{CATALOG_NAME}"),
):
    """
    Perform a cone search on the OpenUniverse2024 Roman/Rubin galaxy_info catalogs.

    Parameters
    ----------
    ra_center : float
        Right Ascension of the GW localization center (degrees, ICRS).
    dec_center : float
        Declination of the GW localization center (degrees, ICRS).
    radius_deg : float
        Search radius in degrees.
    s3_prefix : str, optional
        Root S3 prefix for the OpenUniverse Roman/Rubin catalogs.

    Returns
    -------
    pandas.DataFrame
        Subset of galaxies within the cone, including:
        ['galaxy_id', 'ra', 'dec', 'sep_arcsec'].
    """

    # Step 1. Determine which HEALPix tiles overlap the GW cone
    healpix = ah.HEALPix(nside=32, order="ring")
    pixels = healpix.cone_search_lonlat(
        lon=ra_center, lat=dec_center , radius=radius_deg )
    print(f"Identified {len(pixels)} overlapping HEALPix regions:{pixels}")

    # Step 2. Load only those parquet region files from S3

    # Connect to the public S3 bucket anonymously
    fs = s3fs.S3FileSystem(anon=True)

    region_paths = []
    flux_paths = []

    for pix in pixels:
        region_path = f"{s3_prefix}/galaxy_{pix}.parquet"
        flux_path   = f"{s3_prefix}/galaxy_flux_{pix}.parquet"

        if fs.exists(region_path):
            region_paths.append(region_path)
            flux_paths.append(flux_path)

    # If no files were found, the localization is likely outside the simulation’s sky coverage.
    if not region_paths:
        print(
            "⚠️ No matching region files found — likely the GW localization "
            "is outside the simulated sky area."
        )
        return pd.DataFrame(columns=["galaxy_id", "ra", "dec", "sep_arcsec"])

    df_info = ds.dataset(region_paths, format="parquet", filesystem=fs)
    df_flux = ds.dataset(flux_paths, format="parquet", filesystem=fs)

    df_info = df_info.to_table().to_pandas()
    df_flux = df_flux.to_table().to_pandas()

    #merge info and flux tables
    df_all = pd.merge(df_info, df_flux, on="galaxy_id", how="left")

    # Step 3. Fine-grained cone search on sky coordinates

    # Convert both the catalog and the GW center into SkyCoord objects
    catalog_coords = SkyCoord(df_all["ra"].values * u.deg,
                              df_all["dec"].values * u.deg)
    center = SkyCoord(ra_center, dec_center)

    # Compute angular separation between each galaxy and the GW position
    sep = catalog_coords.separation(center)
    df_all["sep_arcsec"] = sep.arcsec

    # Keep only galaxies within the search radius
    mask = sep <= (radius_deg)
    df_subset = df_all[mask]

    print(f"✅ Found {len(df_subset)} galaxies within {radius_deg:.3f}° "
          f"of RA={ra_center:.3f}, Dec={dec_center:.3f}")

    return df_subset
```

```{code-cell} ipython3
df_candidates = cone_search_catalog(ra_center,dec_center,radius_deg)
```

```{code-cell} ipython3
# Take a look at what we have in the dataframe of candidates.
df_candidates
```

### 3.2 Image access
Now we need to find the filenames of the images in the TDS survey which include these targets

```{code-cell} ipython3
---
jupyter:
  source_hidden: true
---
def TDS_image_search(ra_center, dec_center, radius_deg, bandname,
                    s3_prefix=f"{BUCKET_NAME}/{OU_PREFIX}/{ROMAN_TDS_PREFIX}/"):
    """
    Query OpenUniverse2024 Roman/Rubin TDS images within a given sky radius.

    Parameters
    ----------
    ra_center : float
        Right Ascension of the GW localization center (degrees, ICRS).
    dec_center : float
        Declination of the GW localization center (degrees, ICRS).
    radius_deg : float
        Search radius in degrees.
    bandname : string
        bandname for which to do photometry
    s3_prefix : str, optional
        Root path to the OpenUniverse Roman/Rubin images on S3.

    Returns
    -------
    image_filenames : list (str)
        Should start with the str "s3://"
    """

    # This is a fudging of the image access part so I can keep working
    image_filenames = [
        'nasa-irsa-simulations/openuniverse2024/roman/full/RomanTDS/images/simple_model/J129/10190/Roman_TDS_simple_model_J129_10190_10.fits.gz',
        'nasa-irsa-simulations/openuniverse2024/roman/full/RomanTDS/images/simple_model/J129/10190/Roman_TDS_simple_model_J129_10190_11.fits.gz',
        'nasa-irsa-simulations/openuniverse2024/roman/full/RomanTDS/images/simple_model/J129/10190/Roman_TDS_simple_model_J129_10190_12.fits.gz',
        'nasa-irsa-simulations/openuniverse2024/roman/full/RomanTDS/images/simple_model/J129/10190/Roman_TDS_simple_model_J129_10190_13.fits.gz',
        'nasa-irsa-simulations/openuniverse2024/roman/full/RomanTDS/images/simple_model/J129/10190/Roman_TDS_simple_model_J129_10190_14.fits.gz']

    #make these the correct path with the s3 prepended on the front
    image_filenames = [f"s3://{path}" for path in image_filenames]

    return image_filenames
```

```{code-cell} ipython3
---
jupyter:
  source_hidden: true
---
def add_image_filenames(df_candidates, radius_deg, bandname):
    """
    For each galaxy candidate, use TDS_image_search() to find nearby images
    and store results as a nested column inside `df_candidates`.

    Parameters
    ----------
    df_candidates : pandas.DataFrame
        Must include at least 'ra' and 'dec' columns.
    radius_deg : float
        Search radius in degrees passed to TDS_image_search().
    bandname : string
        name of the filter to use

    Returns
    -------
    pandas.DataFrame
        The same DataFrame that was passed in, now containing an added
        column ``"image_filenames"``. Each entry of this column is a
        list of strings with the S3 paths of overlapping TDS images.
    """

    # Store filenames for each candidate
    filenames_all = []

    for idx, row in df_candidates.iterrows():
        ra, dec = row["ra"], row["dec"]
        filenames = TDS_image_search(ra, dec, radius_deg, bandname)
        filenames_all.append(filenames)

    df_candidates["image_filenames"] = filenames_all

    return df_candidates
```

```{code-cell} ipython3
bandname = "J129"
df_candidates = add_image_filenames(df_candidates, radius_deg, bandname)
```

```{code-cell} ipython3
#note our dataframe now contains a column with a list of filenames per candidate host galaxy
df_candidates
```

## 4.  Make a Light Curve
This section demonstrates how to extract and visualize a light curve for a potential gravitational-wave host galaxy using simulated Roman images. The first function, `run_aperture_photometry()`, performs simple circular aperture photometry on a set of FITS images from S3 using the astropy [photutils](https://photutils.readthedocs.io/en/stable/) package . The second function, `plot_light_curve()`, then compiles these measurements into a time-ordered plot showing how the observed flux evolves across multiple visits, providing a first look at temporal variability that could signal transient activity or host-galaxy changes.

```{code-cell} ipython3
---
jupyter:
  source_hidden: true
---
def run_aperture_photometry(df_candidates, bandname, aperture_radius=3.0):
    """
    Perform circular aperture photometry on a list of FITS images.

    Parameters
    ----------
    df_candidates : pandas.DataFrame
        Must contain columns 'ra', 'dec', and a nested column 'image_filenames'
        (each a list of FITS image paths).
    bandname : string
        Bandname for which to do photometry.
    aperture_radius : float, optional
        Aperture radius in pixels. Default is 3.0.

    Returns
    -------
    phot_df : pandas.DataFrame
        DataFrame containing the photometry results with columns:
        ['RA', 'Dec', 'mjd_obs', 'filename', 'flux', 'flux_err',
         'aperture_radius', 'background']
    """

    #store photometry for all rows in the dataframe
    #these will be lists of lists
    mjd_all, flux_all, flux_err_all = [], [], []

    #for each candidate galaxy:
    for idx, row in df_candidates.iterrows():
        filenames = row["image_filenames"]

        #setup to store for each candidate galaxy
        mjd_list, flux_list, flux_err_list = [], [], []


        for fname in filenames:

            #opening a gzipped fits file, don't recommend changing the next line.
            with fits.open(fname, fsspec_kwargs={"anon": True}, memmap=False) as hdul:
                data = hdul[1].data
                header = hdul[1].header

                # Placeholder pixel coordinates for now
                position = [(40, 40)]

                # Simple circular aperture
                aperture = CircularAperture(position, r=aperture_radius)

                # Perform aperture photometry
                phot_table = aperture_photometry(data, aperture)

                # Check output (optional)
                #print(phot_table)

                # Background estimate (median of finite pixels)
                background = np.nanmedian(data)

                # Subtract background from aperture sum
                flux = phot_table['aperture_sum'][0] - background * np.pi * aperture_radius**2

                # Approximate uncertainty from background rms
                flux_err = np.nanstd(data) * np.sqrt(np.pi * aperture_radius**2)

                # Observation mid-time from MJD-OBS
                mjd_obs = header.get('MJD-OBS', None)

                #store info
                mjd_list.append(mjd_obs)
                flux_list.append(flux)
                flux_err_list.append(flux_err)


        mjd_all.append(mjd_list)
        flux_all.append(flux_list)
        flux_err_all.append(flux_err_list)

    # Add as nested columns
    df_candidates["mjd_obs"] = mjd_all
    df_candidates[f"flux_{bandname}"] = flux_all
    df_candidates[f"flux_err_{bandname}"] = flux_err_all

    return df_candidates
```

```{code-cell} ipython3
df = run_aperture_photometry(df_candidates, bandname, aperture_radius=3.0)
```

```{code-cell} ipython3
#take a quick look at the dataframe of aperture photometry to see what we are working with
df
```

```{code-cell} ipython3
---
jupyter:
  source_hidden: true
---
def plot_single_light_curve(df, galaxy_id, bandname):
    """
    Plot a single galaxy's light curve from a DataFrame, with error bars.

    Parameters
    ----------
    df : pandas.DataFrame
        Must contain columns 'galaxy_id', 'mjd_obs',
        f'flux_{bandname}', and optionally f'flux_err_{bandname}'.
    galaxy_id : int or str
        Galaxy identifier to plot.
    bandname : str
        Photometric band name used for column labels .

    Returns
    -------
    matplotlib.figure.Figure
        The generated figure.
    """
    #make a series for each galaxy
    row = df[df["galaxy_id"] == galaxy_id].squeeze()
    if row.empty:
        print(f"⚠️ Galaxy {galaxy_id} not found in DataFrame.")
        return None

    # Extract nested arrays
    #need to go to numpy so we can check for non-finite values
    times = np.array(row["mjd_obs"], dtype=float)
    fluxes = np.array(row[f"flux_{bandname}"], dtype=float)
    flux_errs = np.array(row[f"flux_err_{bandname}"], dtype=float)

    # Drop NaNs / non-finite values
    mask = np.isfinite(times) & np.isfinite(fluxes)
    mask &= np.isfinite(flux_errs)
    times, fluxes, flux_errs = times[mask], fluxes[mask], flux_errs[mask]
    if len(times) == 0:
        print(f"⚠️ No valid flux points for galaxy {galaxy_id}.")
        return None

    #sort on time
    sort_idx = np.argsort(times)
    times, fluxes, flux_errs = times[sort_idx], fluxes[sort_idx], flux_errs[sort_idx]

    # Plot
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.errorbar(
        times, fluxes, yerr=flux_errs, fmt="o", capsize=3, label=f"Galaxy {galaxy_id}"
    )
    ax.plot(times, fluxes, "-", alpha=0.6, color=ax.get_lines()[-1].get_color())

    ax.set_xlabel("MJD")
    ax.set_ylabel(f"Flux ({bandname})")
    ax.set_title(f"Light Curve for Galaxy {galaxy_id} ({bandname})")
    ax.legend()

    # Restrict x-axis to data range with small padding
    margin = 0.05 * (times.max() - times.min()) if len(times) > 1 else 0.1
    ax.set_xlim(times.min() - margin, times.max() + margin)

    plt.show()
    return fig
```

```{code-cell} ipython3
---
jupyter:
  source_hidden: true
---
def plot_candidate_light_curves(df, bandname):
    """
    Plot all candidate light curves color-coded by galaxy_id,

    Parameters
    ----------
    df : pandas.DataFrame
        Must contain 'galaxy_id', 'mjd_obs', f'flux_{bandname}',
        and optionally f'flux_err_{bandname}'.
    bandname : str
        Photometric band name .

    Returns
    -------
    matplotlib.figure.Figure
        Combined plot figure.
    """
    #setup for plotting
    fig, ax = plt.subplots(figsize=(7, 5))
    colors = itertools.cycle(plt.cm.tab10.colors)

    #for each candidate host galaxy
    for _, row in df.iterrows():
        galaxy_id = row["galaxy_id"]

        # Extract nested arrays
        #need to go to numpy so we can check for non-finite values
        times = np.array(row["mjd_obs"], dtype=float)
        fluxes = np.array(row[f"flux_{bandname}"], dtype=float)
        flux_errs = np.array(row[f"flux_err_{bandname}"], dtype=float)

        # Drop invalid
        mask = np.isfinite(times) & np.isfinite(fluxes)
        mask &= np.isfinite(flux_errs)
        times, fluxes, flux_errs = times[mask], fluxes[mask], flux_errs[mask]

        if len(times) == 0:  #empty photometry
            continue

        #sort on time
        sort_idx = np.argsort(times)
        times, fluxes, flux_errs = times[sort_idx], fluxes[sort_idx], flux_errs[sort_idx]

        #plot
        color = next(colors)
        ax.errorbar(times, fluxes, yerr=flux_errs, fmt="o", capsize=3,
                    color=color, label=str(galaxy_id))
        ax.plot(times, fluxes, "-", color=color, alpha=0.6)

        xmin, xmax = times.min(), times.max()

    ax.set_xlabel("MJD")
    ax.set_ylabel(f"Flux ({bandname})")
    ax.set_title(f"Candidate Light Curves ({bandname})")
    ax.legend(title="Galaxy ID", fontsize="small")

    # Restrict x-axis to data range with padding
    margin = 0.05 * (xmax - xmin) if xmax > xmin else 0.1
    ax.set_xlim(xmin - margin, xmax + margin)

    plt.show()
    return fig
```

```{code-cell} ipython3
#grab one of the galaxy_ids from the printed out dataframe above
favorite = 10306000022321
fig_single = plot_single_light_curve(df, favorite, bandname)
```

This will look better when they are a real time series and not all taken at the same time, also the warning about xlim being set to the same min and max will go away.

```{code-cell} ipython3
fig_all_candidates = plot_candidate_light_curves(df, bandname)
```

all galaxy ids are faked to have the same set of images for photometry hence the points exactly line on top of each other.

+++

## 5. Make Cutouts
We follow the example in this [tutorial](https://caltech-ipac.github.io/irsa-tutorials/openuniverse2024-roman-simulated-timedomainsurvey/) to display cutouts of the potential host galaxies as a function of time with the time listed in the cutout title.

```{code-cell} ipython3
---
jupyter:
  source_hidden: true
---
def make_cutout(fname, ra, dec, size=100):
    """
    Create a sky-aligned cutout around (RA, Dec) from a Roman TDS FITS image on S3.

    Parameters
    ----------
    fname : str
        Full or partial S3 path to the .fits.gz image.
    ra, dec : float
        Target coordinates in degrees.
    size : int or float, optional
        Cutout size in pixels. Default = 100.

    Returns
    -------
    cutout : astropy.nddata.Cutout2D or None
        The rotated image cutout centered on (RA, Dec).
        Returns None if the target is outside the field.
    """

    coord = SkyCoord(ra=ra * u.deg, dec=dec * u.deg)

    with fits.open(fname, fsspec_kwargs={"anon": True}, memmap=False) as hdu:
        img = hdu[1].data
        header = hdu[1].header

        # Get orientation and CD matrix
        CDmat = np.array([
            [header["CD1_1"], header["CD1_2"]],
            [header["CD2_1"], header["CD2_2"]],
        ])
        orientation = header.get("ORIENTAT", 0.0)
        sca = header.get("SCA_ID", 0)

        # Flip chips if needed
        if sca % 3 == 0:
            orientation += 180

        # Rotation matrix (inverse because we rotate image)
        angle = orientation
        rot_img = rotate(img, angle=angle, reshape=False, cval=np.nan)

        # Update header for rotated WCS
        CD1_1_rot = np.cos(-angle * np.pi / 180)
        CD1_2_rot = -np.sin(-angle * np.pi / 180)
        CD2_1_rot = np.sin(-angle * np.pi / 180)
        CD2_2_rot = np.cos(-angle * np.pi / 180)
        RotMat_inv = np.array([[CD1_1_rot, -CD1_2_rot],
                                [-CD2_1_rot, CD2_2_rot]])
        CDmat_rot = np.dot(CDmat, RotMat_inv)
        header["CD1_1"], header["CD1_2"] = CDmat_rot[0]
        header["CD2_1"], header["CD2_2"] = CDmat_rot[1]
        header["ORIENTAT"] -= angle

        rot_wcs = WCS(header)

        #fudge the build cutout for now since we don't have RA and Dec of our sources worked out yet
        position = (40, 40)
        cutout = Cutout2D(img, position, size, mode="partial")

        # Build cutout  (real version)
        #cutout = Cutout2D(rot_img, coord, size, wcs=rot_wcs, mode="partial")

        return cutout
```

```{code-cell} ipython3
---
jupyter:
  source_hidden: true
---
def cutout_gallery(image_filenames, ra, dec, size=100, ncols=4,
                   galaxy_id=None, superevent_id=None):
    """
    Display a gallery of cutouts centered on (RA, Dec) for a list of Roman TDS images.

    Parameters
    ----------
    image_filenames : list of str
        List of S3 image filenames.
    ra, dec : float
        Target coordinates in degrees.
    size : int or float, optional
        Cutout size in pixels. Default = 100.
    ncols : int, optional
        Number of columns in the gallery grid. Default = 4.
    galaxy_id : int or str, optional
        Galaxy ID for labeling the plot.
    superevent_id : str, optional
        GW superevent ID for labeling the plot.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The displayed figure object.
    """
    # Initialize lists to store image cutouts and observation times
    cutouts, mjd_list = [], []

    # Loop over all image filenames and generate cutouts
    for fname in image_filenames:
        cutout = make_cutout(fname, ra, dec, size=size)
        if cutout is not None:
            cutouts.append(cutout.data)
            # Open FITS header to extract observation time (MJD)
            with fits.open(f"s3://{fname}", fsspec_kwargs={"anon": True}, memmap=False) as hdu:
                mjd_list.append(hdu[1].header.get("MJD-OBS", np.nan))

    # Stop if no valid cutouts were created
    if not cutouts:
        raise ValueError("No valid cutouts could be created.")

    # Set up grid dimensions for displaying the gallery
    n_images = len(cutouts)
    nrows = int(np.ceil(n_images / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3 * ncols, 3 * nrows))
    axes = np.atleast_1d(axes).ravel()

    # Build figure title if information is available
    if galaxy_id is not None and superevent_id is not None:
        fig.suptitle(
            f"Cutouts of host galaxy candidate {galaxy_id} for GW event {superevent_id}",
            fontsize=14, y=0.98
        )
    elif galaxy_id is not None:
        fig.suptitle(
            f"Cutouts for candidate galaxy {galaxy_id}",
            fontsize=14, y=0.98
        )

    # Display each cutout image with contrast scaling and MJD label
    for ax, img, mjd, fname in zip(axes, cutouts, mjd_list, image_filenames):
        vmin, vmax = np.nanpercentile(img, [5, 99])
        ax.imshow(img, origin="lower", cmap="gray", vmin=vmin, vmax=vmax)
        ax.set_title(f"MJD {mjd:.2f}", fontsize=8)
        ax.axis("off")

    # Hide any extra axes
    for ax in axes[len(cutouts):]:
        ax.axis("off")

    plt.tight_layout()
    plt.show()
    return
```

```{code-cell} ipython3
# galaxy_id of my favorite candidate
favorite = 10306000022321

single_gal = df[df["galaxy_id"] == favorite]
if single_gal.empty:
    raise ValueError(f"Galaxy {favorite} not found in DataFrame.")

cutout_gallery(
    image_filenames=single_gal["image_filenames"].iloc[0],
    ra=single_gal["ra"].iloc[0],
    dec=single_gal["dec"].iloc[0],
    size=100,
    ncols=3,
    galaxy_id=favorite,
)

# You may get a `FITSFixedWarning` this is completely harmless and
# just means there is an extra space in the DATE-OBS keyword that astropy is fixing.
```

## Acknowledgements

- [IPAC-IRSA](https://irsa.ipac.caltech.edu/)
- This work made use of Astropy:\footnote{http://www.astropy.org} a community-developed core Python package and an ecosystem of tools and resources for astronomy.
- This research made use of Photutils, an Astropy package for
detection and photometry of astronomical sources (Bradley et al.
<2025>).

## About this notebook

**Authors:** IRSA Data Science Team, including Jessica Krick, Jaladh Singhal, Troy Raen, Brigitta Sipőcz,
Andreas Faisst, Vandana Desai

**Updated:** 2025-03-24

**Contact:** [IRSA Helpdesk](https://irsa.ipac.caltech.edu/docs/help_desk.html) with questions
or problems.

**Runtime:** As of the date above, this notebook takes about 10 years to run to completion on
a machine with 8GB RAM and 4 CPU.


**AI Acknowledgement:**

This tutorial was developed with the assistance of OpenAI’s ChatGPT (GPT-5)

**References:**

- Bradley et al., 2025; https://zenodo.org/records/14889440/

- [Robitaille et al., 2013](https://www.aanda.org/articles/aa/full_html/2013/10/aa22068-13/aa22068-13.html)

- [Astropy Collaboration et al., 2018](https://arxiv.org/abs/1801.02634)

- [Astropy Collaboration et al., 2022](https://arxiv.org/abs/2206.14220)

- [Virtanen et al., 2020](https://www.nature.com/articles/s41592-019-0686-2); DOI: 10.1038/s41592-019-0686-2.

- [OpenUniverse et al., 2025](https://arxiv.org/abs/2501.05632)

```{code-cell} ipython3
print("elapsed time", time.time() - starttime)
```
