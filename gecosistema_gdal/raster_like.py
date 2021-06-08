# -------------------------------------------------------------------------------
# Licence:
# Copyright (c) 2012-2021
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
# Name:        rester_like.py
# Purpose:
#
# Author:      Lorenzo Borelli
#
# Created:     31/08/2018
# -------------------------------------------------------------------------------
from osgeo  import gdal,gdalconst
from osgeo import osr, ogr
import numpy as np
from .gdalwarp import *
from .gdal_utils import *


def EquivalentRasterFiles(filename1, filename2):
    """
    EquivalentRasterFiles
    """
    ds1 = gdal.Open(filename1, gdalconst.GA_ReadOnly)
    ds2 = gdal.Open(filename2, gdalconst.GA_ReadOnly)

    if ds1 and ds2:
        arr1 = ds1.GetRasterBand(1).ReadAsArray()
        arr2 = ds2.GetRasterBand(1).ReadAsArray()

        return np.array_equal(arr1, arr2, equal_nan=True) and SameExtent(filename1, filename2) and SamePixelSize(
            filename1, filename2) and SameSpatialRef(filename1, filename2)
    return False


def GetPixelSize(filename):
    """
    GetPixelSize
    """
    dataset = gdal.Open(filename, gdalconst.GA_ReadOnly)
    if dataset:
        gt = dataset.GetGeoTransform()
        _, px, _, _, _, py = gt
        dataset = None
        return (px,abs(py))
    return (0,0)

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

def CreateRectangleShape(minx, miny, maxx, maxy, srs, fileshp="tempxy...."):
    """
    CreateRectangleShape
    """
    fileshp = fileshp if fileshp else "./tempdir/rect.shp"
    # Write rest to Shapefile
    # srs = osr.SpatialReference()
    # srs.ImportFromEPSG(spatialRefSys)
    driver = ogr.GetDriverByName("ESRI Shapefile")
    if os.path.exists(fileshp):
        driver.DeleteDataSource(fileshp)
    ds = driver.CreateDataSource(fileshp)
    layer = ds.CreateLayer(fileshp, srs, geom_type=ogr.wkbPolygon)
    featureDefn = layer.GetLayerDefn()
    feature = ogr.Feature(featureDefn)
    rect = Rectangle(minx, miny, maxx, maxy)
    feature.SetGeometry(rect)
    layer.CreateFeature(feature)
    feature, layer, ds = None, None, None
    return fileshp

def RasterLike(filetif, filetpl, fileout=None, verbose=False):
    """
    RasterLike: adatta un raster al raster template ( dem ) ricampionando, riproiettando estendendo/clippando il file raster se necessario.
    """
    if SameSpatialRef(filetif, filetpl) and SamePixelSize(filetif, filetpl, decimals=2) and SameExtent(filetif, filetpl, decimals=3):
        fileout = filetif
        return fileout

    if verbose:
        # print("SameSpatialRef:",SameSpatialRef(filetif, filetpl))
        # print("SamePixelSize:", SamePixelSize(filetif, filetpl))
        print("SameExtent:", SameExtent(filetif, filetpl))
        print(GetExtent(filetif))
        print(GetExtent(filetpl))

    if verbose:
        print("1)gdalwarp...")
    file_warp1 = gdalwarp([filetif], dstSRS=GetSpatialRef(filetpl), pixelsize=GetPixelSize(filetpl))

    tif_minx, tif_miny, tif_maxx, tif_maxy = GetExtent(file_warp1)
    tpl_minx, tpl_miny, tpl_maxx, tpl_maxy = GetExtent(filetpl)
    # create tif and template rectangles
    # to detect intersections
    tif_rectangle = Rectangle(tif_minx, tif_miny, tif_maxx, tif_maxy)
    tpl_rectangle = Rectangle(tpl_minx, tpl_miny, tpl_maxx, tpl_maxy)

    if verbose:
        print('rectangle done')
    if tif_rectangle.Intersects(tpl_rectangle):
        if verbose:
            print('intersection')
        file_rect = tempfilename(suffix=".shp")
        spatialRefSys = GetSpatialRef(filetpl)
        demshape = CreateRectangleShape(tpl_minx, tpl_miny, tpl_maxx, tpl_maxy,
                                        srs=spatialRefSys,
                                        fileshp=file_rect)
        if verbose:
            print("2)gdalwarp...")
        gdalwarp([file_warp1], fileout, cutline=demshape, cropToCutline=True,
                 dstSRS=GetSpatialRef(filetpl), pixelsize=GetPixelSize(filetpl))

        os.unlink(file_rect)

    else:
        wdata, geotransform, projection = GDAL2Numpy(filetpl, band=1, dtype=np.float32, load_nodata_as=np.nan)
        wdata.fill(np.nan)
        Numpy2GTiff(wdata, geotransform, projection, fileout)

    os.unlink(file_warp1)

    return fileout if os.path.exists(fileout) else None