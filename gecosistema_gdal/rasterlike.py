# -------------------------------------------------------------------------------
# Licence:
# Copyright (c) 2012-2021 Luzzi Valerio
#
# The above copyright notice and this permission notice shall be
# included in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
# OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
# NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
# HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
# WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
# OTHER DEALINGS IN THE SOFTWARE.
#
#
# Name:        rasterlike.py
# Purpose:
#
# Author:      Lorenzo Borelli, Luzzi Valerio
#
# Created:     16/06/2021
# -------------------------------------------------------------------------------
import os
import shutil
import site
import glob
import tempfile
import datetime
import numpy as np
from osgeo import osr, ogr
from osgeo import gdal, gdalconst

def now():
    return datetime.datetime.now()

def total_seconds_from(t):
    return (datetime.datetime.now() - t).total_seconds()

def printf( text, verbose = False):
    """
    printf - print just if verbose
    """
    if verbose:
        print(f"{text}")

def done(text, t, verbose = False):
    """
    done - print the total seconds elapsed from t
    """
    if verbose:
        seconds = total_seconds_from(t)
        print(f"{text} done in {seconds}s.")

def normpath(pathname):
    """
    normpath
    """
    if not pathname:
        return ""
    return os.path.normpath(pathname.replace("\\", "/")).replace("\\", "/")


def justpath(pathname, n=1):
    """
    justpath
    """
    for j in range(n):
        (pathname, _) = os.path.split(normpath(pathname))
    if pathname == "":
        return "."
    return normpath(pathname)


def juststem(pathname):
    """
    juststem
    """
    pathname = os.path.basename(normpath(pathname))
    (root, _) = os.path.splitext(pathname)
    return root


def justext(pathname):
    """
    justext
    """
    pathname = os.path.basename(normpath(pathname))
    (_, ext) = os.path.splitext(pathname)
    return ext.lstrip(".")


def forceext(pathname, newext):
    """
    forceext
    """
    (root, _) = os.path.splitext(normpath(pathname))
    pathname = root + ("." + newext if len(newext.strip()) > 0 else "")
    return normpath(pathname)


def tempfilename(prefix="tmp", suffix=""):
    """
    tempfilename
    """
    return "%s/%s%s%s" % (tempfile.gettempdir(), prefix, datetime.datetime.now().timestamp(), suffix)


def mkdirs(pathname):
    """
    mkdirs - create a folder
    mkdirs("hello/world)
    mkdirs("hello/world/file.tif) #file.tif must exists
    """
    if not os.path.isdir(pathname):
        try:
            if os.path.isfile(pathname):
                pathname = justpath(pathname)
            os.makedirs(pathname)
        except:
            pass
        return os.path.isdir(pathname)
    return True



def __Numpy2GTiff__(arr, geotransform, projection, filename, save_nodata_as=-9999):
    """
    __Numpy2GTiff__
    """
    GDT = {
        'uint8': gdal.GDT_Byte,
        'uint16': gdal.GDT_UInt16,
        'uint32': gdal.GDT_UInt32,
        'int16': gdal.GDT_Int16,
        'int32': gdal.GDT_Int32,

        'float32': gdal.GDT_Float32,
        'float64': gdal.GDT_Float64
    }

    if isinstance(arr, np.ndarray):
        rows, cols = arr.shape
        if rows > 0 and cols > 0:
            dtype = str(arr.dtype).lower()
            fmt = GDT[dtype] if dtype in GDT else gdal.GDT_Float64

            mkdirs(justpath(filename))

            CO = ["BIGTIFF=YES", "TILED=YES", "BLOCKXSIZE=256", "BLOCKYSIZE=256", 'COMPRESS=LZW']
            driver = gdal.GetDriverByName("GTiff")
            dataset = driver.Create(filename, cols, rows, 1, fmt, CO)
            if (geotransform != None):
                dataset.SetGeoTransform(geotransform)
            if (projection != None):
                dataset.SetProjection(projection)
            dataset.GetRasterBand(1).SetNoDataValue(save_nodata_as)
            dataset.GetRasterBand(1).WriteArray(arr)
            dataset = None
            return filename
    return None

def SetGDALEnv():
    """
    SetGDALEnv
    """
    os.environ["__PROJ_LIB__"]  = os.environ["PROJ_LIB"] if "PROJ_LIB" in os.environ else ""
    os.environ["__GDAL_DATA__"] = os.environ["GDAL_DATA"] if "GDAL_DATA" in os.environ else ""
    os.environ["PROJ_LIB"] = find_PROJ_LIB()
    os.environ["GDAL_DATA"] = find_GDAL_DATA()

def RestoreGDALEnv():
    """
    RestoreGDALEnv
    """
    if "__PROJ_LIB__" in os.environ:
        os.environ["PROJ_LIB"] = os.environ["__PROJ_LIB__"]
    if "__GDAL_DATA__" in os.environ:
        os.environ["GDAL_DATA"] = os.environ["__GDAL_DATA__"]

def find_PROJ_LIB():
    """
    find_PROJ_LIB - the path of proj_lib
    """
    pathnames = []
    roots = site.getsitepackages()
    for root in roots:
        pathnames += glob.glob(root + "/osgeo/**/proj.db", recursive=True)
        if len(pathnames):
            break
    return justpath(pathnames[0]) if len(pathnames) else ""


def find_GDAL_DATA():
    """
    find_GDAL_DATA - the path of GDAL_DATA
    """
    pathnames = []
    roots = site.getsitepackages()
    for root in roots:
        pathnames += glob.glob(root + "/osgeo/**/gt_datum.csv", recursive=True)
        if len(pathnames):
            break
    return justpath(pathnames[0]) if len(pathnames) else ""


def gdalwarp(filelist, fileout=None, dstSRS="", cutline="", cropToCutline=False, tap=False, multithread=False, pixelsize=(0, 0), verbose=False):
    """
    gdalwarp
    """
    filelist = [filelist] if isinstance(filelist,str) else filelist
    fileout = fileout if fileout else tempfilename(suffix=".tif")

    kwargs = {
        "format": "GTiff",
        "outputType": gdalconst.GDT_Float32,
        "creationOptions": ["BIGTIFF=YES", "TILED=YES", "BLOCKXSIZE=256", "BLOCKYSIZE=256", "COMPRESS=LZW"],
        "dstNodata": -9999,
        "resampleAlg": gdalconst.GRIORA_Bilinear,
        "multithread": multithread
    }

    if pixelsize[0] > 0 and pixelsize[1] > 0:
        kwargs["xRes"] = pixelsize[0]
        kwargs["yRes"] = pixelsize[1]

    if dstSRS:
        kwargs["dstSRS"] = dstSRS

    if os.path.isfile(cutline):
        kwargs["cropToCutline"] = cropToCutline
        kwargs["cutlineDSName"] = cutline
        kwargs["cutlineLayer"] = juststem(cutline)

    # gdal.Warp depends on PROJ_LIB and GDAL_DATA --------------------------
    # os.environ["PROJ_LIB"] = ..../site-packages/osgeo/data/proj
    # patch PROJ_LIB - save it before and restore after gdalwarp
    PROJ_LIB = os.environ["PROJ_LIB"] if "PROJ_LIB" in os.environ else ""
    GDAL_DATA = os.environ["GDAL_DATA"] if "GDAL_DATA" in os.environ else ""
    # print(find_PROJ_LIB())
    os.environ["PROJ_LIB"] = find_PROJ_LIB()
    # print(find_GDAL_DATA())
    os.environ["GDAL_DATA"] = find_GDAL_DATA()

    gdal.Warp(fileout, filelist, **kwargs)
    if PROJ_LIB:
        os.environ["PROJ_LIB"] = PROJ_LIB
    if GDAL_DATA:
        os.environ["GDAL_DATA"] = GDAL_DATA
    # ----------------------------------------------------------------------
    return fileout


def GetPixelSize(filename):
    """
    GetPixelSize
    """
    dataset = gdal.Open(filename, gdalconst.GA_ReadOnly)
    if dataset:
        gt = dataset.GetGeoTransform()
        _, px, _, _, _, py = gt
        dataset = None
        return px, abs(py)
    return 0, 0


def SamePixelSize(filename1, filename2, decimals=-1):
    """
    SamePixelSize
    """
    size1 = GetPixelSize(filename1)
    size2 = GetPixelSize(filename2)
    if decimals >= 0:
        size1 = [round(item, decimals) for item in size1]
        size2 = [round(item, decimals) for item in size2]
    return size1 == size2


def GetEmptyLike(filename, dtype=np.float32, default=np.nan):
    """
    GetMetadata
    """
    dataset = gdal.Open(filename, gdalconst.GA_ReadOnly)
    if dataset:
        m, n = dataset.RasterYSize, dataset.RasterXSize
        prj = dataset.GetProjection()
        gt = dataset.GetGeoTransform()
        dataset = None
        res = np.empty( (m,n), dtype = dtype)
        res.fill(default)
        return res, gt, prj
    return np.array([np.nan]), None, None


def GetArea(filename):
    """
    GetArea
    """
    dataset = gdal.Open(filename, gdalconst.GA_ReadOnly)
    if dataset:
        m, n = dataset.RasterYSize, dataset.RasterXSize
        gt = dataset.GetGeoTransform()
        _, px, _, _, _, py = gt
        dataset = None
        return m*n*px*abs(py)
    return 0


def GetExtent(filename):
    """
    GetExtent
    """
    ext = justext(filename).lower()
    if ext == "tif":
        dataset = gdal.Open(filename, gdalconst.GA_ReadOnly)
        if dataset:
            "{xmin} {ymin} {xmax} {ymax}"
            m, n = dataset.RasterYSize, dataset.RasterXSize
            gt = dataset.GetGeoTransform()
            xmin, px, _, ymax, _, py = gt
            xmax = xmin + n * px
            ymin = ymax + m * py
            ymin, ymax = min(ymin, ymax), max(ymin, ymax)
            dataset = None
            return xmin, ymin, xmax, ymax

    elif ext in ("shp", "dbf"):

        filename = forceext(filename, "shp")
        driver = ogr.GetDriverByName("ESRI Shapefile")
        dataset = driver.Open(filename, 0)
        if dataset:
            layer = dataset.GetLayer()
            extent = layer.GetExtent()
            dataset = None
            xmin, xmax, ymin, ymax = extent
            return xmin, ymin, xmax, ymax

    return 0, 0, 0, 0


def SameExtent(filename1, filename2, decimals=-1):
    """
    SameExtent
    """
    extent1 = GetExtent(filename1)
    extent2 = GetExtent(filename2)
    if decimals >= 0:
        extent1 = [round(item, decimals) for item in extent1]
        extent2 = [round(item, decimals) for item in extent2]
    return extent1 == extent2


def GetSpatialRef(filename):
    """
    GetSpatialRef
    """
    if isinstance(filename, osr.SpatialReference):
        srs = filename

    elif isinstance(filename, int):
        srs = osr.SpatialReference()
        srs.ImportFromEPSG(filename)

    elif isinstance(filename, str) and filename.lower().startswith("epsg:"):
        code = int(filename.split(":")[1])
        srs = osr.SpatialReference()
        srs.ImportFromEPSG(code)

    elif isinstance(filename, str) and os.path.isfile(filename) and filename.lower().endswith(".shp"):
        ds = ogr.OpenShared(filename)
        if ds:
            srs = ds.GetLayer().GetSpatialRef()
        ds = None

    elif isinstance(filename, str) and os.path.isfile(filename) and filename.lower().endswith(".tif"):
        ds = gdal.Open(filename, gdalconst.GA_ReadOnly)
        if ds:
            wkt = ds.GetProjection()
            srs = osr.SpatialReference()
            srs.ImportFromWkt(wkt)
        ds = None
    else:
        srs = osr.SpatialReference()
    return srs


def SameSpatialRef(filename1, filename2):
    """
    SameSpatialRef
    """
    srs1 = GetSpatialRef(filename1)
    srs2 = GetSpatialRef(filename2)
    if srs1 and srs2:
        return srs1.IsSame(srs2)
    return None


def Rectangle(minx, miny, maxx, maxy):
    """
    Rectangle
    """
    ring = ogr.Geometry(ogr.wkbLinearRing)
    ring.AddPoint_2D(minx, miny)
    ring.AddPoint_2D(maxx, miny)
    ring.AddPoint_2D(maxx, maxy)
    ring.AddPoint_2D(minx, maxy)
    ring.AddPoint_2D(minx, miny)
    # Create polygon
    poly = ogr.Geometry(ogr.wkbPolygon)
    poly.AddGeometry(ring)
    return poly


def RectangleFromFileExtent(filename):
    """
    RectangleFromFileExtent
    """
    minx, miny, maxx, maxy = GetExtent(filename)
    return Rectangle(minx, miny, maxx, maxy) if minx else None


def ShapeExtentFrom(filetif, fileshp=""):
    """
    ShapeExtentFrom
    """
    fileshp = fileshp if fileshp else tempfilename(prefix="rect_",suffix=".shp")
    # Write rest to Shapefile
    driver = ogr.GetDriverByName("ESRI Shapefile")
    if os.path.exists(fileshp):
        driver.DeleteDataSource(fileshp)
    ds = driver.CreateDataSource(fileshp)
    layer = ds.CreateLayer(fileshp, srs=GetSpatialRef(filetif), geom_type=ogr.wkbPolygon)
    featureDefn = layer.GetLayerDefn()
    feature = ogr.Feature(featureDefn)
    rect = RectangleFromFileExtent(filetif)
    feature.SetGeometry(rect)
    layer.CreateFeature(feature)
    feature, layer, ds = None, None, None
    return fileshp

def RasterLike(filetif, filetpl, fileout=None, verbose=False):
    """
    RasterLike: adatta un raster al raster template ( dem ) ricampionando, riproiettando estendendo/clippando il file raster se necessario.
    """
    t0 = now()
    if SameSpatialRef(filetif, filetpl) and SamePixelSize(filetif, filetpl, decimals=2) and SameExtent(filetif, filetpl, decimals=3):

        fileout = fileout if fileout else tempfilename(suffix=".tif")
        if fileout != filetif:
            #Copy the file with the fileout name
            shutil.copy2(filetif, fileout)
            return fileout

        return filetif

    # Special case where filetif is bigger than filetpl so we make crop first an then resampling
    if GetArea(filetif) >= 4 * GetArea(filetpl):
        #1) Crop
        printf("1) Crop...",verbose)
        file_rect = ShapeExtentFrom(filetpl)
        file_warp1 = gdalwarp(filetif, cutline=file_rect, cropToCutline=True, dstSRS=GetSpatialRef(filetpl))
        #2) Resampling and refine the extent
        printf("2) Resampling...",verbose)
        fileout = gdalwarp(file_warp1, fileout, pixelsize=GetPixelSize(filetpl), cutline=file_rect, cropToCutline=True)
        os.unlink(file_warp1)
        os.unlink(file_rect)
        done("gdalwarp",t0,verbose)
        return fileout

    printf("1) gdalwarp for resampling...",verbose)
    file_warp1 = gdalwarp(filetif, dstSRS=GetSpatialRef(filetpl), pixelsize=GetPixelSize(filetpl))

    #tif_minx, tif_miny, tif_maxx, tif_maxy = GetExtent(file_warp1)
    #tpl_minx, tpl_miny, tpl_maxx, tpl_maxy = GetExtent(filetpl)

    ## create tif and template rectangles
    ## to detect intersections
    #tif_rectangle = Rectangle(tif_minx, tif_miny, tif_maxx, tif_maxy)
    #tpl_rectangle = Rectangle(tpl_minx, tpl_miny, tpl_maxx, tpl_maxy)

    tif_rectangle = RectangleFromFileExtent(file_warp1)
    tpl_rectangle = RectangleFromFileExtent(filetpl)

    if tif_rectangle.Intersects(tpl_rectangle):
        #file_rect = tempfilename(suffix=".shp")
        #spatialRefSys = GetSpatialRef(filetpl)
        #file_rect = CreateRectangleShape(tpl_minx, tpl_miny, tpl_maxx, tpl_maxy,srs=spatialRefSys,fileshp=file_rect)
        file_rect = ShapeExtentFrom(filetpl)

        printf("2) gdalwarp for crop...",verbose)
        gdalwarp(file_warp1, fileout, cutline=file_rect, cropToCutline=True,
                 dstSRS=GetSpatialRef(filetpl), pixelsize=GetPixelSize(filetpl))

        os.unlink(file_rect)
    else:
        #GDAL2Numpy cause access to disk
        #wdata, gt, prj = GDAL2Numpy(filetpl, band=1, dtype=np.float32, load_nodata_as=np.nan)
        #wdata.fill(np.nan)
        #__Numpy2GTiff__(wdata, gt, prj, fileout)
        wdata, gt, prj = GetEmptyLike(filetpl)
        __Numpy2GTiff__(wdata, gt, prj, fileout)

    os.unlink(file_warp1)
    done("gdalwarp",t0,verbose)
    return fileout if os.path.exists(fileout) else None
