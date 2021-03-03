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
# Name:        gdal_numpy.py
# Purpose:     Parallel gdal version for numpy
#
# Author:      Luzzi Valerio
#
# Created:     03/08/2017
# -------------------------------------------------------------------------------
import numpy as np

import datetime
import threading
from osgeo import gdal
from gecosistema_core import *

def GDAL_blocksize(filename):
    """
    GDAL_like
    """
    dataset = gdal.Open(filename, gdal.GA_ReadOnly)
    if dataset:
        M, N = int(dataset.RasterYSize), int(dataset.RasterXSize)
        BSx, BSy = dataset.GetRasterBand(1).GetBlockSize()
        Nb = N / BSx + (0 if N % BSx == 0 else 1)
        Mb = M / BSy + (0 if M % BSy == 0 else 1)
        return (BSx, BSy, Mb, Nb, M, N)
    return (-1, -1, -1, -1, -1, -1)


def GDAL_like(filename, fileout=""):
    """
    GDAL_like
    """
    BSx, BSy, Mb, Nb, M, N = 0,0, 0,0, 0,0
    dataset1 = gdal.Open(filename, gdal.GA_ReadOnly)
    dataset2 = None
    if dataset1:
        band1 = dataset1.GetRasterBand(1)
        M, N = int(dataset1.RasterYSize), int(dataset1.RasterXSize)
        B = dataset1.RasterCount
        BSx, BSy = band1.GetBlockSize()
        Nb = int(N / BSx) + (0 if N % BSx == 0 else 1)
        Mb = int(M / BSy) + (0 if M % BSy == 0 else 1)
        CO = ["BIGTIFF=YES"]
        options = dataset1.GetMetadata("IMAGE_STRUCTURE")
        if BSy > 1:
            CO += ["TILED=YES", "BLOCKXSIZE=%d" % BSx, "BLOCKYSIZE=%d" % BSy]
        for key in options:
            if key == "COMPRESSION":
                CO.append("COMPRESS=" + options[key])
            else:
                CO.append(key + "=" + options[key])

        driver = gdal.GetDriverByName("GTiff")
        fileout = fileout if fileout else forceext(filename, "copy.tif")
        dataset2 = driver.Create(fileout, N, M, B, band1.DataType, CO)
        dataset2.SetProjection(dataset1.GetProjection())
        dataset2.SetGeoTransform(dataset1.GetGeoTransform())
        for j in range(1, B + 1):
            band1 = dataset1.GetRasterBand(j)
            band2 = dataset2.GetRasterBand(j)
            if band1.GetNoDataValue() != None:
                band2.SetNoDataValue(band1.GetNoDataValue())
            else:
                band2.SetNoDataValue(np.nan)
    dataset1 = None

    return (dataset2, BSx, BSy, Mb, Nb, M, N)

def GDALReadBlock(dataset, blocno, BSx=-1, BSy=-1, verbose=False):
    """
    GDALReadBlock
    """
    dataset = gdal.Open(dataset, gdal.GA_ReadOnly) if isstring(dataset) else dataset
    if dataset:
        band = dataset.GetRasterBand(1)

        BSx, BSy = (BSx, BSy) if BSx > 0 else band.GetBlockSize()
        M, N = int(dataset.RasterYSize), int(dataset.RasterXSize)

        Nb = int(N / BSx) + (0 if N % BSx == 0 else 1)

        x0 = (blocno % Nb) * BSx
        y0 = int(blocno / Nb) * BSy

        QSx = BSx if x0 + BSx <= N else N % BSx
        QSy = BSy if y0 + BSy <= M else M % BSy

        data = band.ReadAsArray(x0, y0, QSx, QSy)

        # Manage No Data
        nodata = band.GetNoDataValue()
        bandtype = gdal.GetDataTypeName(band.DataType)
        if bandtype in ('Byte', 'Int16', 'Int32', 'UInt16', 'UInt32', 'CInt16', 'CInt32'):
            data = data.astype("Float32", copy=False)
        if bandtype in ('Float32', 'Float64', 'CFloat32', 'CFloat64'):
            data[data == nodata] = np.nan

        if verbose and blocno % 100 == 0:
            sys.stdout.write('r')
        if isstring(dataset):
            dataset, band = None, None
        return data
    return None


def GDALWriteBlock(dataset, blocno, data, BSx=-1, BSy=-1, verbose=False):
    """
    GDALWriteBlock
    """
    dataset = gdal.Open(dataset, gdal.GA_Update) if isstring(dataset) else dataset
    if dataset:
        band = dataset.GetRasterBand(1)
        BSx, BSy = (BSx, BSy) if BSx > 0 else band.GetBlockSize()

        M, N = int(dataset.RasterYSize), int(dataset.RasterXSize)
        Nb = int(N / BSx) + (0 if N % BSx == 0 else 1)

        yoff = int(blocno / Nb) * BSy
        xoff = (blocno % Nb) * BSx

        band.WriteArray(data, xoff, yoff)

        if verbose and blocno % 100 == 0:
            sys.stdout.write('w')
        if isstring(dataset):
            dataset, band = None, None

        return True

    return False


def GDALExpressionBlock(blocno, expression, env, BSx=-1, BSy=-1):
    """
    GDALExprBlock
    """
    # load block into numpy variables
    z = env["__out__"]
    keys = env.keys()
    for varname in keys:
        if varname.startswith("file_") and varname.replace("file_", "") in re.split(r'[^0-9a-zA-Z_]+', expression):
            name = varname.replace("file_", "")
            env[name] = GDALReadBlock(env[varname], blocno, BSx, BSy)

    data = eval(expression, env)
    GDALWriteBlock(env["file_" + z], blocno, data, BSx, BSy)
    data = None


def gdal_numpy(expression, env={}, ignore_warn=True, verbose=False):
    """
    gdal_numpy

    expression = "z = a+b"
    env = {
        "a":"fileA.tif",
        "b":"fileB.tif",
        "z":"fileout.tif"
    }
    """
    if ignore_warn:
        np.seterr(all='ignore')

    # could be usefull
    expression = sformat(expression, env)

    z, expr = expression.split("=", 1)
    z, expr = z.strip(), expr.strip()
    env["__out__"] = z

    # import numpy by default
    env["np"] = np
    env["numpy"] = np

    t1 = datetime.datetime.now()

    # variables
    rvars = re.split(r'[^0-9a-zA-Z_]+', expr)
    rvarnames = [varname for varname in env.keys() if
                 isstring(env[varname]) and file(env[varname]) and (varname in rvars)]

    # pre-open all files
    afilename = z
    for varname in rvarnames:
        if varname != z:
            # print("loading %s readonly"%varname)
            env["file_" + varname] = gdal.Open(env[varname], gdal.GA_ReadOnly)
            afilename = env[varname]

    if afilename:

        if z in rvarnames:
            # print("loading %s readwrite" % z)
            env["file_" + z] = gdal.Open(env[z], gdal.GA_Update)
            BSx, BSy, Mb, Nb, M, N = GDAL_blocksize(env[z])
        else:
            # print("creating new %s" % z)
            (env["file_" + z], BSx, BSy, Mb, Nb, M, N) = GDAL_like(afilename, env[z])

        NB = Nb * Mb
        for j in range(NB):
            GDALExpressionBlock(j, expr, env, BSx, BSy)

        keys = env.keys()
        for key in keys:
            if key.startswith("file_"):
                env[key] = None

        t2 = datetime.datetime.now()
        if verbose:
            print("\n<%s> done in %ss." % (expression, (t2 - t1).total_seconds()))
    else:
        raise Exception(
            "Expression is not valid. Give an array on right part of the expression to correctly detect the size.")

    return env[z]



