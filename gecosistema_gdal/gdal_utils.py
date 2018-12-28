# -------------------------------------------------------------------------------
# Licence:
# Copyright (c) 2012-2018 Luzzi Valerio 
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
# Name:        gdal_utils.py
# Purpose:
#
# Author:      Luzzi Valerio
#
# Created:     31/08/2018
# -------------------------------------------------------------------------------
import gdal,gdalconst
import numpy as np
import struct
from gecosistema_core import *


def MapToPixel(mx,my,gt):
    '''
    MapToPixel - Convert map to pixel coordinates
    @param  mx:    Input map x coordinate (double)
    @param  my:    Input map y coordinate (double)
    @param  gt:    Input geotransform (six doubles)
    @return: px,py Output coordinates (two ints)
    '''
    if gt[2]+gt[4]==0: #Simple calc, no inversion required
        px = (mx - gt[0]) / gt[1]
        py = (my - gt[3]) / gt[5]

        return int(px),int(py)

    raise Exception("I need to Invert geotransform!")

#-------------------------------------------------------------------------------
#   GetValueAt
#-------------------------------------------------------------------------------
def GetValueAt(X,Y,filename):
    """
    GetValueAt -
    """
    #Converto in epsg 3857
    dataset = gdal.Open(filename,gdalconst.GA_ReadOnly)
    if dataset:
        band = dataset.GetRasterBand(1)
        cols = dataset.RasterXSize
        rows = dataset.RasterYSize
        gt = dataset.GetGeoTransform()
        #Convert from map to pixel coordinates.
        #Only works for geotransforms with no rotation.
        #If raster is rotated, see http://code.google.com/p/metageta/source/browse/trunk/metageta/geometry.py#493
        #print  gt
        j,i = MapToPixel(float(X),float(Y),gt)

        if i in range(0,band.YSize) and j in range(0,band.XSize):
            scanline=  band.ReadRaster(j,i,1,1,buf_type= gdalconst.GDT_Float32) #Assumes 16 bit int aka 'short'
            (value,) = struct.unpack('f' , scanline)
            return value

    #raise ValueError("Unexpected (Lon,Lat) values.")
    return None

def GDAL2Numpy(pathname, band=1):
    """
    GDAL2Numpy
    """
    dataset = gdal.Open(pathname, gdalconst.GA_ReadOnly)
    if dataset:
        band = dataset.GetRasterBand(band)
        cols = dataset.RasterXSize
        rows = dataset.RasterYSize
        geotransform = dataset.GetGeoTransform()
        projection = dataset.GetProjection()
        nodata = band.GetNoDataValue()
        bandtype = gdal.GetDataTypeName(band.DataType)
        wdata = band.ReadAsArray(0, 0, cols, rows)
        # translate nodata as Nan
        if not wdata is None:
            if bandtype in ('Float32', 'Float64', 'CFloat32', 'CFloat64'):
                if not nodata is None and abs(nodata) > 3.4e38:
                    wdata[abs(wdata) > 3.4e38] = np.nan
                elif not nodata is None:
                    wdata[wdata == nodata] = np.nan
            elif bandtype in ('Byte', 'Int16', 'Int32', 'UInt16', 'UInt32', 'CInt16', 'CInt32'):
                wdata = wdata.astype("Float32", copy=False)
                wdata[wdata == nodata] = np.nan
        band = None
        dataset = None
        return (wdata, geotransform, projection)
    print("file %s not exists!" % (pathname))
    return (None, None, None)

def gdal_translate(src_dataset, dst_dataset=None, of="GTiff", ot="Float32", xres=-1, yres=-1, compress=True,
                   verbose=False):
    """
    gdal_translate -q -of GTiff -ot Float32 -tr 25 25 "{src_dataset}" "{dst_dataset}"
    """
    translate_inplace = False
    command = """gdal_translate -q -of {of} -ot {ot} """
    command += """-tr {xres} {yres} """ if xres > 0 and yres > 0 else ""
    command += """--config GDAL_CACHEMAX 90% """
    command += """-co "BIGTIFF=YES" -co "TILED=YES" -co "BLOCKXSIZE=256" -co "BLOCKYSIZE=256" -co "BIGTIFF=YES" """
    command += """-co "COMPRESS=LZW" -co "PREDICTOR={predictor}" """
    command += """"{src_dataset}" "{dst_dataset}" """

    if ot in ("Float32", "Float64", "CFloat32", "CFloat64"):
        predictor = 3
    elif ot in ("Int16", "UInt16", "Int32", "UInt32", "CInt16", "CInt32"):
        predictor = 2
    else:
        predictor = 1

    if not dst_dataset: # or samepath(src_dataset, dst_dataset):
        translate_inplace = True
        dst_dataset = justpath(src_dataset) + "/" + tempname("tmp_")

    env = {

        "src_dataset": src_dataset,
        "dst_dataset": dst_dataset,
        "ot": ot,
        "of": of,
        "xres": xres,
        "yres": yres,
        "predictor": predictor
    }

    if Exec(command, env, precond=[src_dataset], postcond=[dst_dataset], skipIfExists=False, verbose=verbose):

        if translate_inplace:
            remove(src_dataset)
            rename(dst_dataset, src_dataset)
            dst_dataset = src_dataset

        return dst_dataset
    return False

def gdalwarp(src_dataset, dst_dataset="", cutline="", of="GTiff", xres=-1, yres=-1, interpolation="bilinear", t_srs="",
             compress="", extraparams="", verbose=False):
    """
    gdalwarp -q -multi -cutline "{fileshp}" -crop_to_cutline -tr {pixelsize} {pixelsize} -of GTiff "{src_dataset}" "{dst_dataset}"
    """

    command  = """gdalwarp -multi -overwrite -q -of {of} """
    command += """-dstnodata -9999 """
    command += """-co "BIGTIFF=YES" -co "TILED=YES" -co "BLOCKXSIZE=256" -co "BLOCKYSIZE=256" """
    command += """--config GDAL_CACHEMAX 90% -wm 500 """
    command += """-cutline "{cutline}" -crop_to_cutline """ if cutline else ""
    command += """-tr {xres} -{yres} """ if xres > 0 and yres > 0 else ""
    command += """-r {interpolation} """
    command += """-t_srs {t_srs} """ if t_srs else ""
    command += """"{src_dataset}" "{dst_dataset}" """
    command += """{extraparams}"""

    translate_inplace = False
    if not dst_dataset:# or samepath(src_dataset, dst_dataset):
        translate_inplace = True
        dst_dataset = justpath(src_dataset) + "/" + tempname("tmp_")

    env = {
        "cutline": cutline,
        "src_dataset": src_dataset,
        "dst_dataset": dst_dataset,
        "of": of,
        "xres": xres,
        "yres": yres,
        "interpolation": interpolation,
        "t_srs": t_srs,
        "compress": compress,
        "extraparams": extraparams
    }

    #Exec(command, env, precond=[src_dataset], postcond=[dst_dataset], skipIfExists=False, verbose=verbose):
    dst_dataset = Exec(command, env, precond=[src_dataset], postcond=[dst_dataset], skipIfExists=True,
                          verbose=verbose)

    if translate_inplace:
        remove(src_dataset)
        rename(dst_dataset, src_dataset)
        dst_dataset = src_dataset

    if compress:
        gdal_translate(dst_dataset, dst_dataset, of, "Float32", xres, yres, compress=True, verbose=verbose)

    return dst_dataset