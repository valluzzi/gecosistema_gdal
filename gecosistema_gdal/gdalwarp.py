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
# Name:        gdalwarp.py
# Purpose:
#
# Author:      Luzzi Valerio
#
# Created:     23/04/2021
# -------------------------------------------------------------------------------
import os,site,glob
from osgeo import gdal,gdalconst
from gecosistema_core import *


def find_PROJ_LIB():
    """
    find_PROJ_LIB - the path of proj_lib
    """
    pathnames = []
    roots = site.getsitepackages()
    for root in roots:
        pathnames+=glob.glob(root+"/osgeo/**/proj.db",recursive=True)
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
        pathnames+=glob.glob(root+"/osgeo/**/gt_datum.csv",recursive=True)
        if len(pathnames):
            break
    return justpath(pathnames[0]) if len(pathnames) else ""

def gdalwarp(filelist, fileout, dstSRS="", cutline="", cropToCutline=False, pixelsize=(0, 0)):
    """
    gdalwarp
    """

    kwargs = {
        "format": "GTiff",
        "outputType": gdalconst.GDT_Float32,
        "creationOptions": ["BIGTIFF=YES", "TILED=YES", "BLOCKXSIZE=256", "BLOCKYSIZE=256", "COMPRESS=LZW"],
        "dstNodata": -9999,
        "resampleAlg": gdalconst.GRIORA_Bilinear,
        "multithread": False
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

    # cache = gdal.GetCacheMax()
    # gdal.SetCacheMax(14000)
    # print(gdal.GetCacheMax())
    gdal.Warp(fileout, filelist, **kwargs)
    if PROJ_LIB:
        os.environ["PROJ_LIB"] = PROJ_LIB
    if GDAL_DATA:
        os.environ["GDAL_DATA"] = GDAL_DATA
    # ----------------------------------------------------------------------
    return fileout