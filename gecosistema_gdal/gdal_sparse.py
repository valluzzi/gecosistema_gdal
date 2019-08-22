# -------------------------------------------------------------------------------
# Licence:
# Copyright (c) 2012-2019 Luzzi Valerio 
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
# Name:        gdal_sparse.py
# Purpose:
#
# Author:      Luzzi Valerio
#
# Created:     24/07/2019
# -------------------------------------------------------------------------------
import osr
import numpy as np
import gdal,gdalconst

from scipy.sparse import issparse


def Sparse2Raster(arr, x0, y0, epsg, px, py, filename="", save_nodata_as=-9999):
    """
    Sparse2Rastersave_nodata_as
    """
    BS = 256
    geotransform = (x0, px, 0.0, y0, 0.0, -(abs(py)))
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(int("%s" % (epsg)))
    projection = srs.ExportToWkt()
    if issparse(arr):
        m, n = arr.shape
        if m > 0 and n > 0:
            dtype = str(arr.dtype)
            if dtype in ["uint8"]:
                fmt = gdal.GDT_Byte
            elif dtype in ["uint16"]:
                fmt = gdal.GDT_UInt16
            elif dtype in ["uint32"]:
                fmt = gdal.GDT_UInt32
            elif dtype in ["float32"]:
                fmt = gdal.GDT_Float32
            elif dtype in ["float64"]:
                fmt = gdal.GDT_Float64
            else:
                fmt = gdal.GDT_Float64

            CO = ["BIGTIFF=YES", "TILED=YES", "BLOCKXSIZE=256", "BLOCKYSIZE=256", 'COMPRESS=LZW']
            driver = gdal.GetDriverByName("GTiff")
            dataset = driver.Create(filename, n, m, 1, fmt, CO)
            if (geotransform != None):
                dataset.SetGeoTransform(geotransform)
            if (projection != None):
                dataset.SetProjection(projection)

            band = dataset.GetRasterBand(1)
            band.SetNoDataValue(save_nodata_as)

            for i in range(0, m, BS):
                for j in range(0, n, BS):
                    BY = min(m - i, BS)
                    BX = min(n - j, BS)
                    a = arr[i:i + BY, j:j + BX].todense()
                    if save_nodata_as==0 and (np.isnan(a)).all():
                        #do nothing
                        pass
                    else:
                        band.WriteArray(a, j, i)

            dataset = None
            return filename
    return None