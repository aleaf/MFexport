import os
from pathlib import Path
import time
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio import Affine
from shapely.geometry import LineString
import matplotlib.pyplot as plt
from gisutils import df2shp


def export_array(filename, a, modelgrid, nodata=-9999,
                 fieldname='value', verbose=False,
                 **kwargs):
    """
    Write a numpy array to Arc Ascii grid or shapefile with the model
    reference.

    Parameters
    ----------
    modelgrid : Flopy StructuredGrid instance
    filename : str
        Path of output file. Export format is determined by
        file extention.
        '.asc'  Arc Ascii grid
        '.tif'  GeoTIFF (requries rasterio package)
        '.shp'  Shapefile
    a : 2D or 3D numpy.ndarray
        Array to export
    nodata : scalar
        Value to assign to np.nan entries (default -9999)
    fieldname : str
        Attribute field name for array values (shapefile export only).
        (default 'values')
    kwargs:
        keyword arguments to np.savetxt (ascii)
        rasterio.open (GeoTIFF)
        or flopy.export.shapefile_utils.write_grid_shapefile2

    Notes
    -----
    Rotated grids will be either be unrotated prior to export,
    using scipy.ndimage.rotate (Arc Ascii format) or rotation will be
    included in their transform property (GeoTiff format). In either case
    the pixels will be displayed in the (unrotated) projected geographic
    coordinate system, so the pixels will no longer align exactly with the
    model grid (as displayed from a shapefile, for example). A key difference
    between Arc Ascii and GeoTiff (besides disk usage) is that the
    unrotated Arc Ascii will have a different grid size, whereas the GeoTiff
    will have the same number of rows and pixels as the original.

    """
    a = a.copy()
    t0 = time.time()
    filename = Path(filename)
    if filename.suffix == ".tif":
        if len(np.unique(modelgrid.delr)) != len(np.unique(modelgrid.delc)) != 1 \
                or modelgrid.delr[0] != modelgrid.delc[0]:
            raise ValueError('GeoTIFF export require a uniform grid.')

        x0 = modelgrid.xyedges[0][0]
        y0 = modelgrid.xyedges[1][0]
        xul, yul = modelgrid.get_coords(x0, y0)
        trans = Affine(modelgrid.delr[0], 0., xul,
                  0., -modelgrid.delc[0], yul) * \
                      Affine.rotation(-modelgrid.angrot)

        # third dimension is the number of bands
        if len(a.shape) == 2:
            a = np.reshape(a, (1, a.shape[0], a.shape[1]))

        if a.dtype == np.int64:
            a = a.astype(np.int32)

        if a.dtype == bool:
            a = a.astype(np.int32)
        meta = {'count': a.shape[0],
                'width': a.shape[2],
                'height': a.shape[1],
                'nodata': nodata,
                'dtype': a.dtype,
                'driver': 'GTiff',
                'crs': modelgrid.crs,
                'transform': trans,
                'compress': 'lzw'
                }
        meta.update(kwargs)

        # remove file first,
        # to avoid rasterio._err.CPLE_AppDefinedError: TIFFReadDirectory: errors
        for file in [filename, filename.with_suffix('.tif.msk'), filename.with_suffix('.aux.xml')]:
            file.unlink(missing_ok=True)
        with rasterio.open(filename, 'w', **meta) as dst:
            dst.write(a)
            if isinstance(a, np.ma.masked_array):
                dst.write_mask(~a.mask.transpose(1, 2, 0))
        print('wrote {}'.format(filename))

    elif filename.lower().endswith(".shp"):
        raise NotImplementedError()
    if verbose:
        print("array export took {:.2f}s".format(time.time() - t0))


def export_array_contours(filename, a, modelgrid,
                          fieldname='level',
                          interval=None,
                          levels=None,
                          maxlevels=1000,
                          crs=None, verbose=False,
                          **kwargs):
    """
    Contour an array using matplotlib; write shapefile of contours.

    Parameters
    ----------
    filename : str
        Path of output file with '.shp' extention.
    a : 2D numpy array
        Array to contour
    crs : obj
        A Python int, dict, str, or :class:`pyproj.crs.CRS` instance
        passed to the :meth:`pyproj.crs.CRS.from_user_input`
        See http://pyproj4.github.io/pyproj/stable/api/crs/crs.html#pyproj.crs.CRS.from_user_input.
        Can be any of:
        
          - PROJ string
          - Dictionary of PROJ parameters
          - PROJ keyword arguments for parameters
          - JSON string with PROJ parameters
          - CRS WKT string
          - An authority string [i.e. 'epsg:4326']
          - An EPSG integer code [i.e. 4326]
          - A tuple of ("auth_name": "auth_code") [i.e ('epsg', '4326')]
          - An object with a `to_wkt` method.
          - A :class:`pyproj.crs.CRS` class
    **kwargs : keyword arguments to matplotlib.axes.Axes.contour

    """
    t0 = time.time()
    if crs is None:
        crs = modelgrid.crs

    if interval is not None:
        kwargs['levels'] = make_levels(a, interval, maxlevels)
    elif levels is not None:
        kwargs['levels'] = levels

    ax = plt.subplots()[-1]
    contours = ax.contour(modelgrid.xcellcenters,
                          modelgrid.ycellcenters,
                          a, **kwargs)
    #plt.savefig('junk.pdf')
    plt.close()

    if not isinstance(contours, list):
        contours = [contours]

    geoms = []
    level = []
    for ctr in contours:
        levels = ctr.levels
        #for i, c in enumerate(ctr.collections):
        #    paths = c.get_paths()
        paths = ctr.get_paths()
        for i, path in enumerate(paths):
            # break the paths up into their components
            # (so that different instances of a contour level 
            # don't connect across other contour levels)
            if len(path) > 0:
                parts = np.split(path.vertices, 
                                np.where(path.codes == 1)[0], axis=0)
                parts = [p for p in parts if len(p) > 0]
                geoms += [LineString(p) if len(p) > 1 else LineString() for p in parts]
                level += list(np.ones(len(parts)) * levels[i])
                #geoms += [LineString(p.vertices) if len(p) > 1 else LineString() for p in paths]
                #level += list(np.ones(len(paths)) * levels[i])

    # convert the dictionary to a recarray
    if len(level) == 0:
        print('No contours! Try adjusting the levels or interval.')
        return
    gdf = gpd.GeoDataFrame({'level': level, 'geometry': geoms}, crs=crs)
    gdf.to_file(filename)
    if verbose:
        print("array contour export took {:.2f}s".format(time.time() - t0))
    return


def make_levels(array, interval, maxlevels=1000):
    imin = np.round(np.floor(np.nanmin(array)), 0)
    imax = np.round(np.ceil(np.nanmax(array)), 0)
    levels = np.round(np.arange(imin, imax, interval), 6)
    inrange = (levels >= np.nanmin(array)) & (levels <= np.nanmax(array))
    levels = levels[inrange]
    if len(levels) > maxlevels:
        msg = '{:.0f} levels at interval of {}; setting contours based on maxlevels ({})'.format(
            len(levels),
            interval,
            maxlevels)
        print(msg)
        levels = np.round(np.linspace(imin, imax, maxlevels), 6)
    return levels


def squeeze_3d(array):
    """Squeeze a 3D array to only include the (2D) slices
    along the 0 axis that are different (for example, periods when
    a stress changes). Include the first slice (period) by default.

    Parameters
    ----------
    array : 3D numpy array
        Original data

    Returns
    -------
    squeezed : dict
        Dictionary of the 2D slices (values), keyed by period, that are
        different.

    """
    unique_pers = list(np.where(np.diff(array, axis=0).sum(axis=(1, 2)) != 0)[0] + 1)
    # include first period by default,
    # won't show up if second is the same
    unique_pers = [0] + unique_pers
    squeezed = {per: array[per] for per in unique_pers}
    return squeezed