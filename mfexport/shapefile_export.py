import time
import numpy as np
import pandas as pd
import geopandas as gpd
from packaging import version
from shapely.geometry import Polygon, Point
import flopy
from flopy.utils import MfList
from flopy.mf6.data.mfdatalist import MFTransientList
try:
    from flopy.mf6.data.mfdataplist import MFPandasTransientList
except:
    MFPandasTransientList = False
from gisutils import df2shp
from mfexport.list_export import mftransientlist_to_dataframe


def export_shapefile(filename, data, modelgrid, kper=None,
                     cellid_col='cellid',
                     squeeze=True, crs=None, geom_type='polygon',
                     epsg=None, proj_str=None, prj=None,
                     verbose=False, **kwargs):
    t0 = time.time()
    if isinstance(data, MFTransientList) or isinstance(data, MfList):
        df = mftransientlist_to_dataframe(data, squeeze=squeeze)
    elif MFPandasTransientList and isinstance(data, MFPandasTransientList):
        df = mftransientlist_to_dataframe(data, squeeze=squeeze)
    elif isinstance(data, np.recarray):
        df = pd.DataFrame(data)
    elif isinstance(data, pd.DataFrame):
        df = data
    else:
        raise TypeError("data needs to be a pandas DataFrame, MFList, or numpy recarray")
    if kper is not None:
        df = df.loc[df['per'] == kper]

    if epsg is not None:
        raise ValueError("export_shapefile: 'epsg' arg is deprecated, use 'crs' instead")
    elif proj_str is not None:
        raise ValueError("export_shapefile: 'proj_str' arg is deprecated, use 'crs' instead, "
                         "and consider using an EPSG code or WKT string instead of a PROJ string.")
    elif prj is not None:
        with open(prj) as src:
            crs = src.read()
    else:
        crs = modelgrid.crs
    
    if 'geometry' not in df.columns:
        i, j = None, None
        if cellid_col in df.columns:
            if isinstance(df[cellid_col].values[0], tuple):
                k, i, j = list(zip(*df[cellid_col]))
                # unfortunately, reaches through inactive cells 
                # lose their cellid (k, i, j) location
                # so there is no way to plot these 
                # without geometries from another source (such as the sfrlines)
                # drop such geometries, which are identified by k, i, j == -1
                invalid_geoms = (np.array(i) < 0) | (np.array(j) < 0)
                df = df.loc[~invalid_geoms].copy()
                i = np.array(i)
                j = np.array(j)
            else:
                raise NotImplementedError(
                    f"shapefile_export: cellid_col '{cellid_col}' column. "
                    "Exporting feature locations from 1-D node numbers not supported.\n"
                    "Supply a geometry column or MAW connectiondata input with k, i, j locations.")
                #invalid_geoms = df[cellid_col] < 0
                #df = df.loc[~invalid_geoms].copy()
                #i = df[cellid_col]
        elif 'i' in df.columns and 'j' in df.columns:
            invalid_geoms = np.any(df[['i', 'j']] < 0, axis=1)
            df = df.loc[~invalid_geoms].copy()
            i, j = df['i'].values, df['j'].values
        else:
            raise ValueError(f"shapefile_export: No cellid column '{cellid_col}' or "
                             "i, j columns in input dataframe.")

        if geom_type.lower() == 'polygon':
            if i is not None and j is not None:
                verts = np.array(modelgrid._cell_vert_list(i, j))
                polys = np.array([Polygon(v) for v in verts])
            elif j is None:
                # TODO: unstructured grid support
                # need to translate 3D node numbers into 2D node numbers
                # otherwise these calls result in out of bounds errors
                verts = [modelgrid.get_cell_vertices(nn) for nn in df[cellid_col]]
                polys = [Polygon(v) for v in verts]
            df['geometry'] = polys
        elif geom_type.lower() == 'point':
            if i is not None and j is not None:
                points = [Point(x, y) for x, y in zip(modelgrid.xcellcenters[i, j],
                                                    modelgrid.ycellcenters[i, j])]
            elif j is None:
                # TODO: unstructured grid support
                points = [Point(x, y) for x, y in zip(modelgrid.xcellcenters.ravel()[i],
                                    modelgrid.ycellcenters.ravel()[i])]
            df['geometry'] = points
        else:
            raise ValueError(f'shapefile_export: unrecognized geom_type: {geom_type}')

    gdf = gpd.GeoDataFrame(df, crs=crs)
    gdf.to_file(filename, **kwargs)
    print(f"wrote {filename}")
    if verbose:
        print("shapefile export took {:.2f}s".format(time.time() - t0))
