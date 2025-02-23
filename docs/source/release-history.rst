===============
Release History
===============

Version 0.3.0 (2024-01-12)
----------------------------
* add support for zone budget output to listfile module; fix issue that occured when there is only one stress period
* add suport for regular Flopy StructuredGrids
* read_sfr_output: add support for using an 'sfrlines' shapefile produced by SFRmaker in lieu Modflow 6 package data
* fixes to cell budget output processing to support multiple instances of a package in MODFLOW 6
* various other fixes to support pandas 2.0 and flopy 3.5

Version 0.2.0 (2022-08-01)
----------------------------
* added option to ``listfile.plot_budget_summary`` to plot annual budget sums from MODFLOW Listing file
* updated ``MFexportGrid`` object to use the :class:`pyproj.crs.CRS` object for coordinate reference system management; added dataframe and grid cell polygon properties
* added online documentation
* added plotting MODFLOW listing file budgets
* added support for SFR input export (MF6 only)
* added option to heads export to export depth to water and overpressurization depth
* bug fixes:
  * skip MODFLOW 6 perioddata (transientlist datatype) for now; flopy array access can be too slow for large datasets
  * various other bug fixes

Initial Release Version 0.1.0 (2019-11-17)
-------------------------------------------
* PDF and gis (raster and shapefile) export for MODFLOW-NWT and MODFLOW-6 style models at model, package and variable levels
* model summary tables via the mfexport.summarize() method
* export of cell budget, head and drawdown information to gis file formats (PDF not implemented yet), for MODFLOW-NWT or MODFLOW-6 style models
* export of SFR package results for MODFLOW-NWT or MODFLOW-6 style models to gis file formats and PDFs
* only uniform structured grids are supported at this time

