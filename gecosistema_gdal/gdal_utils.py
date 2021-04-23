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
import os
from osgeo import osr,ogr
from osgeo import gdal,gdalconst
import numpy as np
import struct
import glob
import site
import tempfile
from gecosistema_core import *
from gdal2numpy import GDAL2Numpy,Numpy2GTiff
from .gdalwarp import *



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
        return (px,abs(py))
    return (0,0)

def SamePixelSize(filename1, filename2):
    """
    SamePixelSize
    """
    px1, py1 = GetPixelSize(filename1)
    px2, py2 = GetPixelSize(filename2)
    return px1 == px2 and py1 == py2

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
    ext = justext(filename).lower()
    if ext =="tif":
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

    elif ext in ("shp","dbf"):

        filename = forceext(filename,"shp")
        driver = ogr.GetDriverByName("ESRI Shapefile")
        dataset = driver.Open(filename, 0)
        if dataset:
            layer = dataset.GetLayer()
            extent = layer.GetExtent()
            dataset = None
            xmin, xmax, ymin, ymax = extent
            return (xmin, ymin, xmax, ymax )

    return (0,0,0,0)

def SameExtent(filename1, filename2):
    """
    SameExtent
    """
    minx1, miny1, maxx1, maxy1 = GetExtent(filename1)
    minx2, miny2, maxx2, maxy2 = GetExtent(filename2)
    return minx1 == minx2 and miny1 == miny2 and maxx1 == maxx2 and maxy1 == maxy2

def GetSpatialReference(filename):
    """
    GetSpatialReference
    """
    dataset = gdal.Open(filename, gdalconst.GA_ReadOnly)
    if dataset:
       return dataset.GetProjection()
    return None

def GetSpatialRef(filename):
    """
    GetSpatialRef
    """
    if isinstance(filename,osr.SpatialReference):
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
        ds= None
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

def Numpy2Raster(arr, x0, y0, epsg, px, py, filename="", save_nodata_as=-9999):
    """
    Numpy2Raster
    """
    gt = (x0, px, 0.0, y0, 0.0, -(abs(py)) )
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(int("%s"%(epsg)))
    prj = srs.ExportToWkt()
    return Numpy2GTiff(arr, gt, prj, filename, save_nodata_as)

def GDAL_LZW( filename,  fileout="", save_nodata_as=-9999 ):
    """
    GDAL LZW compression
    """
    filelzw = fileout if fileout else forceext(filename,"lzw.tif")
    data, gt, prj = GDAL2Numpy(filename, load_nodata_as=save_nodata_as )
    Numpy2GTiff(data, gt, prj, filelzw,  save_nodata_as=save_nodata_as)
    if fileout=="" and isfile(filelzw):
        rename(filelzw, filename, overwrite=True)
        return filename
    return filelzw

def GDALError( filenameA, filenameB, file_err):
    """
    GDALError - return a raster = filenameA-filenameB
    shape and projection must be the same
    """
    file_err = file_err if file_err else "err.tif"
    data1, _, _ = GDAL2Numpy(filenameA, dtype=np.Float32, load_nodata_as=0.0)
    data2, _, _ = GDAL2Numpy(filenameB, dtype=np.Float32, load_nodata_as=0.0)

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

    Numpy2GTiff(out_array.astype("uint8"), gt, prj, dst_dataset, no_data)
    out_array=None
    return True

def gdal_translate(src_dataset, dst_dataset=None, of="GTiff", ot="Float32", xres=-1, yres=-1, extraparams="", verbose=False):
    """
    gdal_translate -q -of GTiff -ot Float32 -tr 25 25 "{src_dataset}" "{dst_dataset}"
    """
    translate_inplace = False
    command = """gdal_translate -q -of {of} -ot {ot} """
    command += """-tr {xres} {yres} """ if xres > 0 and yres > 0 else ""
    #command += """--config GDAL_CACHEMAX 90% """
    command += """-co "BIGTIFF=YES" -co "TILED=YES" -co "BLOCKXSIZE=256" -co "BLOCKYSIZE=256" """
    command += """-co "COMPRESS=LZW" """
    command += """"{src_dataset}" "{dst_dataset}" """
    command += """{extraparams}"""

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
        "extraparams":extraparams
    }

    if Exec(command, env, precond=[src_dataset], postcond=[dst_dataset], skipIfExists=False, verbose=verbose):

        if translate_inplace:
            remove(src_dataset)
            rename(dst_dataset, src_dataset)
            dst_dataset = src_dataset

        return dst_dataset
    return False

def gdal_rasterize(fileshp, filetpl, fileout=None):
    """
    gdal_rasterize
    """
    fileout = fileout if fileout else fileshp.replace(".shp", ".tif")
    creation_options = ["BIGTIFF=YES", "TILED=YES", "BLOCKXSIZE=256", "BLOCKYSIZE=256", 'COMPRESS=LZW']
    options = []

    ds = gdal.Open(filetpl, 0)
    gt = ds.GetGeoTransform()
    m, n = ds.RasterYSize, ds.RasterXSize

    xmin, px, _, ymax, _, py = gt
    xmax = xmin + px * n
    ymin = ymax + py * m

    # read source vector
    vector = ogr.Open(fileshp, 0)
    layer = vector.GetLayer()
    target_ds = gdal.GetDriverByName('GTiff').Create(fileout, n, m, 1, gdal.GDT_Byte, creation_options)
    target_ds.SetGeoTransform((xmin, px, 0, ymax, 0, py))

    target_ds.SetProjection(layer.GetSpatialRef().ExportToWkt())
    band = target_ds.GetRasterBand(1)
    band.SetNoDataValue(0)

    gdal.RasterizeLayer(target_ds, [1], layer, burn_values=[1])

    band.FlushCache()

    ds = None
    band = None
    target_ds = None
    return fileout if isfile(fileout) else False

def gdal_crop(src_dataset, dst_dataset, cutline, nodata=-9999, xres=-1, yres=-1, interpolation="nearest", extraparams="", verbose=False):
    """
    gdal_translate -q -of GTiff -ot Float32 -tr 25 25 "{src_dataset}" "{dst_dataset}"
    """
    (xmin, ymin, xmax, ymax ) = GetExtent(cutline)
    command = """gdal_translate -q -of GTiff -ot Float32 """
    command += """-tr {xres} -{yres} """ if xres > 0 and yres > 0 else ""
    command += """-r {interpolation} """
    command += """-projwin {xmin} {ymax} {xmax} {ymin} """
    command += """-co "BIGTIFF=YES" -co "TILED=YES" -co "BLOCKXSIZE=256" -co "BLOCKYSIZE=256" """
    command += """-co "COMPRESS=LZW" """
    command += """"{src_dataset}" "{dst_dataset}" """
    command += """{extraparams}"""

    filecrop = tempfname("crop",ext="tif")

    env = {

        "src_dataset": src_dataset,
        "dst_dataset": filecrop,
        "xres": xres,
        "yres": yres,
        "interpolation": interpolation,
        "xmin":xmin,
        "xmax":xmax,
        "ymin":ymin,
        "ymax":ymax,
        "extraparams":extraparams
    }

    if Exec(command, env, precond=[src_dataset], postcond=[filecrop], skipIfExists=False, verbose=verbose):
        filemask = tempfname("mask",ext="tif")
        gdal_rasterize( cutline, filecrop, filemask )
        if filemask:
            mask, _  ,  _  = GDAL2Numpy( filemask, dtype = np.uint8 ,load_nodata_as = 0)
            data, gt , prj = GDAL2Numpy( filecrop, load_nodata_as = np.nan)
            data[mask == 0] = np.nan
            Numpy2GTiff(data, gt, prj, dst_dataset, save_nodata_as= nodata )
            remove( filemask )
        remove( filecrop)
    return dst_dataset

def gdalwarp_exe(src_dataset, dst_dataset="", cutline="", of="GTiff", nodata=-9999, xres=-1, yres=-1, interpolation="near", t_srs="",extraparams="", verbose=False):
    """
    gdalwarp -q -multi -cutline "{fileshp}" -crop_to_cutline -tr {pixelsize} {pixelsize} -of GTiff "{src_dataset}" "{dst_dataset}"
    """

    command  = """gdalwarp -multi -overwrite -q -of {of} """
    command += """-dstnodata {nodata} """
    command += """-co "BIGTIFF=YES" -co "TILED=YES" -co "BLOCKXSIZE=256" -co "BLOCKYSIZE=256" """
    command += """-co "COMPRESS=LZW" """
    command += """--config GDAL_CACHEMAX 90% -wm 500 """
    if isfile(cutline) and lower(justext(cutline)) == "shp":
        command += """-cutline "{cutline}" """
    elif isfile(cutline) and lower(justext(cutline)) == "tif":
        command += """-te {xmin} {ymin} {xmax} {ymax} """
    elif isstring(cutline) and len(listify(cutline))==4:
        command += """-te {xmin} {ymin} {xmax} {ymax} """

    command += """-tr {xres} -{yres} """ if xres > 0 and yres > 0 else ""
    command += """-r {interpolation} """
    command += """-t_srs {t_srs} """ if t_srs else ""
    command += """"{src_dataset}" "{dst_dataset}" """
    command += """{extraparams}"""

    #translate_inplace = False
    #if not dst_dataset:# or samepath(src_dataset, dst_dataset):
    #    translate_inplace = True
    #    dst_dataset = justpath(src_dataset) + "/" + tempname("tmp_")

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

    #if translate_inplace:
    #    remove(src_dataset)
    #    rename(dst_dataset, src_dataset)
    #    dst_dataset = src_dataset

    #if compress:
    #    gdal_translate(dst_dataset, dst_dataset, of, "Float32", xres, yres, compress=True, verbose=verbose)

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

def Rectangle( minx, miny, maxx, maxy ):
    """
    Rectangle
    """
    ring = ogr.Geometry(ogr.wkbLinearRing)
    ring.AddPoint(minx, miny)
    ring.AddPoint(maxx, miny)
    ring.AddPoint(maxx, maxy)
    ring.AddPoint(minx, maxy)
    ring.AddPoint(minx, miny)
    # Create polygon
    poly = ogr.Geometry(ogr.wkbPolygon)
    poly.AddGeometry(ring)
    return poly

def CreateRectangleShape( minx, miny, maxx, maxy, srs,fileshp="tempxy...."):
    """
    CreateRectangleShape
    """
    fileshp = fileshp if fileshp else "./tempdir/rect.shp"
    # Write rest to Shapefile
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

def RasterLike(filetif, filetpl, fileout=None):
    """
    RasterLike: adatta un raster al raster template ( dem ) ricampionando, riproiettando estendendo/clippando il file raster se necessario.
    """
    if SameSpatialRef(filetif, filetpl) and SamePixelSize(filetif, filetpl) and SameExtent(filetif, filetpl):
        fileout = filetif
        return fileout

    gdalwarp([filetif], fileout, dstSRS=GetSpatialRef(filetpl), pixelsize=GetPixelSize(filetpl)[0])

    tif_minx, tif_miny, tif_maxx, tif_maxy = GetExtent(filetif)
    tpl_minx, tpl_miny, tpl_maxx, tpl_maxy = GetExtent(filetpl)

    # create tif and template rectangles
    # to detect intersections
    tif_rectangle = Rectangle(tif_minx, tif_miny, tif_maxx, tif_maxy)
    tpl_rectangle = Rectangle(tpl_minx, tpl_miny, tpl_maxx, tpl_maxy)

    if tif_rectangle.Intersects(tpl_rectangle):
        with tempfile.NamedTemporaryFile(suffix='.shp') as fp:
            spatialRefSys = GetSpatialRef(filetpl)
            demshape = CreateRectangleShape(tpl_minx, tpl_miny, tpl_maxx, tpl_maxy,
                                            srs = spatialRefSys,
                                            fileshp=fp.name)
            gdalwarp([filetif], fileout, cutline=demshape, cropToCutline=True)


    else:
        wdata, geotransform, projection = GDAL2Numpy(filetpl, dtype=np.float32, load_nodata_as=np.nan)
        wdata.fill(np.nan)
        Numpy2GTiff(wdata, geotransform, projection, fileout)

    #if SameSpatialRef(fileout, filetpl) and SamePixelSize(fileout, filetpl) and SameExtent(fileout, filetpl):
    #    print('RasterLike successful')

    return fileout if os.path.exists(fileout) else None



