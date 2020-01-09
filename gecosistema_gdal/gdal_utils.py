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
import osr
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

def GetPixelSize(filename):
    """
    GetPixelSize
    """
    dataset = gdal.Open(filename, gdalconst.GA_ReadOnly)
    if dataset:
        gt = dataset.GetGeoTransform()
        _, px, _, _, _, py = gt
        dataset = None
        return (px,py)
    return (0,0)

def GetRasterShape(filename):
    """
    GetRasterShape
    """
    dataset = gdal.Open(filename, gdalconst.GA_ReadOnly)
    if dataset:
        band = dataset.GetRasterBand(1)
        m,n = dataset.RasterYSize,dataset.RasterXSize
        return (m,n)
    return (0,0)

def GetExtent(filename):
    """
    GetExtent
    """
    dataset = gdal.Open(filename, gdalconst.GA_ReadOnly)
    if dataset:
        "{xmin} {ymin} {xmax} {ymax}"
        m,n  = dataset.RasterYSize,dataset.RasterXSize
        gt = dataset.GetGeoTransform()
        xmin,px,_,ymax,_,py = gt
        xmax = xmin + n*px
        ymin = ymax + m*py
        ymin,ymax = min(ymin,ymax),max(ymin,ymax)
        dataset=None
        return (xmin, ymin, xmax, ymax )
    return (0,0,0,0)

def GetSpatialReference(filename):
    """
    GetSpatialReference
    """
    dataset = gdal.Open(filename, gdalconst.GA_ReadOnly)
    if dataset:
       return dataset.GetProjection()
    return None

def GetNoData(filename):
    """
    GetNoData
    """
    dataset = gdal.Open(filename, gdalconst.GA_ReadOnly)
    if dataset:
        band = dataset.GetRasterBand(1)
        nodata = band.GetNoDataValue()
        data, band, dataset = None, None, None
        return nodata
    return None

def SetNoData(filename, nodata):
    """
    SetNoData
    """
    dataset = gdal.Open(filename, gdalconst.GA_Update)
    if dataset:
        band = dataset.GetRasterBand(1)
        nodata = band.SetNoDataValue(nodata)
        data, band, dataset = None, None, None
    return None

def GDAL2Numpy(pathname, band=1, dtype='', load_nodata_as = np.nan):
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

            # Output datatype
            if dtype and dtype != bandtype:
                wdata = wdata.astype(dtype, copy=False)

            if bandtype in ('Float32', 'Float64', 'CFloat32', 'CFloat64'):
                if not nodata is None and abs(nodata) > 3.4e38:
                    wdata[abs(wdata) > 3.4e38] = load_nodata_as
                elif not nodata is None:
                    wdata[wdata == nodata] = load_nodata_as
            elif bandtype in ('Byte', 'Int16', 'Int32', 'UInt16', 'UInt32', 'CInt16', 'CInt32'):
                #wdata = wdata.astype("Float32", copy=False)
                if nodata != load_nodata_as:
                    wdata[wdata == nodata] = load_nodata_as

        band = None
        dataset = None
        return (wdata, geotransform, projection)
    print("file %s not exists!" % (pathname))
    return (None, None, None)

def Numpy2GTiff(arr, geotransform, projection, filename, save_nodata_as=-9999):
    """
    Numpy2GTiff
    """
    if isinstance(arr, np.ndarray):
        rows, cols = arr.shape
        if rows > 0 and cols > 0:
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
            dataset = driver.Create(filename, cols, rows, 1, fmt, CO)
            if (geotransform != None):
                dataset.SetGeoTransform(geotransform)
            if (projection != None):
                dataset.SetProjection(projection)
            dataset.GetRasterBand(1).SetNoDataValue(save_nodata_as)
            dataset.GetRasterBand(1).WriteArray(arr)
            # ?dataset.GetRasterBand(1).ComputeStatistics(0)
            dataset = None
            return filename
    return None


def Numpy2AAIGrid(data, geotransform, filename, save_nodata_as=-9999):
    """
    Numpy2AAIGrid
    """
    (x0, pixelXSize, rot, y0, rot, pixelYSize) = geotransform
    (rows, cols) = data.shape
    stream = open(filename, "wb")
    stream.write("ncols         %d\r\n" % (cols))
    stream.write("nrows         %d\r\n" % (rows))
    stream.write("xllcorner     %d\r\n" % (x0))
    stream.write("yllcorner     %d\r\n" % (y0 + pixelYSize * rows))
    stream.write("cellsize      %d\r\n" % (pixelXSize))
    stream.write("NODATA_value  %d\r\n" % (save_nodata_as))
    template = ("%.7g " * cols) + "\r\n"
    for row in data:
        line = template % tuple(row.tolist())
        stream.write(line)
    stream.close()
    return filename

def Numpy2Gdal(data, geotransform, projection, filename, save_nodata_as=-9999):
    """
    Numpy2Gdal
    """
    ext = os.path.splitext(filename)[1][1:].strip().lower()
    mkdirs(justpath(filename))
    if ext == "tif" or ext == "tiff":
        return Numpy2GTiff(data, geotransform, projection, filename, save_nodata_as)
    elif ext == "asc":
        return Numpy2AAIGrid(data, geotransform, filename, save_nodata_as)
    else:
        return ""

def Numpy2Raster(arr, x0, y0, epsg, px, py, filename="", save_nodata_as=-9999):
    """
    Numpy2Raster
    """
    gt = (x0, px, 0.0, y0, 0.0, -(abs(py)) )
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(int("%s"%(epsg)))
    prj = srs.ExportToWkt()
    return Numpy2Gdal(arr, gt, prj, filename, save_nodata_as)


def GDALError( filenameA, filenameB, file_err):
    """
    GDALError - return a raster = filenameA-filenameB
    shape and projection must be the same
    """
    file_err = file_err if file_err else "err.tif"
    data1, _, _ = GDAL2Numpy(filenameA, dtype="Float32", load_nodata_as=0.0)
    data2, _, _ = GDAL2Numpy(filenameB, dtype="Float32", load_nodata_as=0.0)

    Numpy2GTiff( data1-data2, gt, prj, file_err, save_nodata_as=0.0)


def gdal_Buffer(src_dataset, dst_dataset=None, distance=10, verbose=True):
    """
    Create a Raster fixed distance buffer
    """
    #hard inspired from
    #https://gis.stackexchange.com/questions/250555/buffering-around-raster-using-gdal-and-numpy
    dst_dataset = dst_dataset if dst_dataset else forceext(src_dataset,"buffer.%sm.tif"%distance)

    ds = gdal.Open(src_dataset)
    if ds is None:
        print("gdal_Buffer error: File <%s> does not exits! " %src_dataset)
        return False
    prj,gt = ds.GetProjection(), ds.GetGeoTransform()
    m,n = ds.RasterYSize,ds.RasterXSize
    band= ds.GetRasterBand(1)
    no_data = band.GetNoDataValue()
    data = band.ReadAsArray(0, 0, n, m).astype(int)
    px = int(abs(gt[1]))
    py = int(abs(gt[5]))
    cell_size = (px + py) / 2
    cell_dist = distance / cell_size
    data[data == (no_data or 0 or -9999)] = 0
    out_array  = np.zeros_like(data)
    temp_array = np.zeros_like(data)
    i, j, h, k = 0, 0, 0, 0

    while (h < n):
        k = 0
        while (k < m):
            if (data[k][h] >= 1):
                i = h - cell_dist
                while ((i < cell_dist + h) and i < n):
                    j = k - cell_dist
                    while (j < (cell_dist + k) and j < m):
                        if (((i - h) ** 2 + (j - k) ** 2) <= cell_dist ** 2):
                            if (temp_array[j][i] == 0 or temp_array[j][i] > ((i - h) ** 2 + (j - k) ** 2)):
                                out_array[j][i] = data[k][h]
                                temp_array[j][i] = (i - h) ** 2 + (j - k) ** 2
                        j += 1
                    i += 1
            k += 1
        h += 1
    ds, temp_array, data = None, None, None

    Numpy2Gdal(out_array.astype("uint8"), gt, prj, dst_dataset, no_data)
    out_array=None
    return True










def gdal_translate(src_dataset, dst_dataset=None, of="GTiff", ot="Float32", xres=-1, yres=-1, compress=True,
                   verbose=False):
    """
    gdal_translate -q -of GTiff -ot Float32 -tr 25 25 "{src_dataset}" "{dst_dataset}"
    """
    translate_inplace = False
    command = """gdal_translate -q -of {of} -ot {ot} """
    command += """-tr {xres} {yres} """ if xres > 0 and yres > 0 else ""
    #command += """--config GDAL_CACHEMAX 90% """
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

def gdalwarp(src_dataset, dst_dataset="", cutline="", of="GTiff", nodata=-9999, xres=-1, yres=-1, interpolation="bilinear", t_srs="",
             compress="", extraparams="", verbose=False):
    """
    gdalwarp -q -multi -cutline "{fileshp}" -crop_to_cutline -tr {pixelsize} {pixelsize} -of GTiff "{src_dataset}" "{dst_dataset}"
    """

    command  = """gdalwarp -multi -overwrite -q -of {of} """
    command += """-dstnodata {nodata} """
    command += """-co "BIGTIFF=YES" -co "TILED=YES" -co "BLOCKXSIZE=256" -co "BLOCKYSIZE=256" """
    command += """--config GDAL_CACHEMAX 90% -wm 500 """
    if isfile(cutline) and lower(justext(cutline)) == "shp":
        command += """-cutline "{cutline}" -crop_to_cutline -tap """
    elif isfile(cutline) and lower(justext(cutline)) == "tif":
        command += """-te {xmin} {ymin} {xmax} {ymax} -tap """
    elif isstring(cutline) and len(listify(cutline))==4:
        command += """-te {xmin} {ymin} {xmax} {ymax} -tap """

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
        "nodata": nodata,
        "xres": xres,
        "yres": yres,
        "interpolation": interpolation,
        "t_srs": t_srs,
        "compress": compress,
        "extraparams": extraparams
    }

    if isfile(cutline) and justext(cutline) == "tif":
        xmin,ymin,xmax,ymax = GetExtent(cutline)
        env["xmin"]=xmin
        env["ymin"]=ymin
        env["xmax"]=xmax
        env["ymax"]=ymax
    elif isstring(cutline) and len(listify(cutline))==4:
        xmin, ymin, xmax, ymax = listify(cutline)
        env["xmin"] = xmin
        env["ymin"] = ymin
        env["xmax"] = xmax
        env["ymax"] = ymax

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

def gdal_merge(workdir, fileout, ignore_value=0, no_data=0, ot="Float32", GDAL_HOME="c:\\Program Files\\GDAL", verbose=False):
    """
    gdal_merge
    """
    filelist   = tempfname("merge",ext="lst")
    filemosaic = fileout

    if ot in ("Float32", "Float64", "CFloat32", "CFloat64"):
        predictor = 3
    elif ot in ("Int16", "UInt16", "Int32", "UInt32", "CInt16", "CInt32"):
        predictor = 2
    else:
        predictor = 1

    env = {
        "GDAL_HOME" :GDAL_HOME,
        "filelist": filelist,
        "filemosaic": filemosaic,
        "fileout": fileout,
        "workdir":workdir,
        "ignore_value":ignore_value,
        "no_data":no_data,
        "ot":ot,
        "predictor":predictor
    }

    with open(filelist,"w+",encoding='utf-8') as stream:
        for filename in ls( workdir, filter =r'.*\.tif'):
            stream.write( filename+"\n")

    command="""python "{GDAL_HOME}\\gdal_merge.py" -n {ignore_value} -a_nodata {no_data} -ot {ot} -of GTiff -co "COMPRESS=LZW"  -co "BIGTIFF=YES" -co "TILED=YES" -co "BLOCKXSIZE=256" -co "BLOCKYSIZE=256" -o "{filemosaic}" --optfile "{filelist}" """

    return Exec(command, env, precond=[], postcond=[filemosaic], remove=[filelist], skipIfExists=False, verbose=verbose)


def ogr2ogr(fileshp, fileout="", format="sqlite", verbose=False):
    """
    ogr2ogr
    ogr2ogr -f "sqlite" output.sqlite  input.shp
    """
    fileout = fileout if fileout else forceext(fileshp,"sqlite")
    command = """ogr2ogr -skipfailures -overwrite -f "{format}" "{fileout}" "{fileshp}" """
    env = {"format":format,"fileshp":fileshp,"fileout":fileout}

    return Exec(command, env, precond=[], postcond=[fileout], skipIfExists=False, verbose=verbose)


def gdal_rasterize(fileshp, snap_to, fileout="",  verbose=False):
    """
    gdal_rasterize
    """
    fileout  = fileout if fileout  else forceext(fileshp,"tif")
    filesnap = snap_to if snap_to else forceext(fileshp,"tif")

    (xmin, ymin, xmax, ymax) = GetExtent(filesnap)
    (px,py) =GetPixelSize(filesnap)

    command = """gdal_rasterize -burn 1 -te {xmin} {ymin} {xmax} {ymax} -tr {px} {py} -tap -ot Byte -a_nodata 255 -of GTiff -l {layername} "{fileshp}" "{fileout}" """
    env = {
        "format":format,
        "fileshp":fileshp,
        "layername":juststem(fileshp),
        "fileout":fileout,
        "xmin":xmin,
        "ymin":ymin,
        "xmax":xmax,
        "ymax":ymax,
        "px":px,
        "py":py
    }

    return Exec(command, env, precond=[], postcond=[fileout], skipIfExists=False, verbose=verbose)

def gdal_contour(filesrc, filedest=None, step=0.0, verbose=False):
    """
    gdal_contour
    """
    filedest = filedest if filedest else forcext(filesrc, "shp")
    if file(filesrc):

        if step<=0.0:
            dataset,_,_ = GDAL2Numpy(filesrc)
            minValue,maxValue=np.min(dataset),np.max(dataset)
            dataset=None
            step = (maxValue-minValue)/20.0

        mkdirs(justpath(filedest))
        command = """gdal_contour -a ELEV -i {step} "{filesrc}" "{filedest}" """
        env = {
            "step":step,
            "filesrc":filesrc,
            "fildest":filedest
        }

        return Exec(command, env, precond=[filesrc], postcond=[filedest], skipIfExists=False, verbose=verbose)

    return False