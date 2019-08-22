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
# Name:        ogr_utils.py
# Purpose:
#
# Author:      Luzzi Valerio
#
# Created:     28/12/2018
# -------------------------------------------------------------------------------
import os,sys,ogr
import gdal,gdalconst

def GetFeatures(fileshp):
    """
    GetFeatures
    """
    res = []
    dataset = ogr.OpenShared(fileshp)
    if dataset:
        layer = dataset.GetLayer(0)
        for feature in layer:
            res.append(feature)
    dataset = None
    return res


def GetFeatureByFid(fileshp, layername=0, fid=0):
    """
    GetFeatureByFid
    """
    feature = None
    dataset = ogr.OpenShared(fileshp)
    if dataset:
        layer = dataset.GetLayer(layername)
        feature = layer.GetFeature(fid)
    dataset = None
    return feature


def GetFeatureBy(fileshp, layername=0, attrname="ogr_id", attrvalue=0):
    """
    GetFeatureByAttr - get the first feature with attrname=attrvalue
    """
    feature = None
    dataset = ogr.OpenShared(fileshp)
    if dataset:
        layer = dataset.GetLayer(layername)
        layerDefinition = layer.GetLayerDefn()
        fieldnames = [layerDefinition.GetFieldDefn(j).GetName().lower() for j in range(layerDefinition.GetFieldCount())]
        if attrname.lower() in fieldnames:
            for feature in layer:
                if feature.GetField(attrname) == attrvalue:
                    dataset = None
                    return feature

    dataset = None
    return None

def GetAttributeTableByFid(fileshp, layername=0, fid=0):
    """
    GetAttributeTableByFid
    """
    res = {}
    dataset = ogr.OpenShared(fileshp)
    if dataset:
        layer = dataset.GetLayer(layername)
        feature = layer.GetFeature(fid)
        geom = feature.GetGeometryRef()
        res["geometry"] = geom.ExportToWkt()
        layerDefinition = layer.GetLayerDefn()
        for j in range(layerDefinition.GetFieldCount()):
            fieldname = layerDefinition.GetFieldDefn(j).GetName()
            res[fieldname] = feature.GetField(j)
    dataset = None
    return res


def removeShape(filename):
    """
    removeShape
    """
    try:
        if file(filename):
            driver = ogr.GetDriverByName('ESRI Shapefile')
            driver.DeleteDataSource(filename)
    except Exception as ex:
        print(ex)
        return None


def SaveFeature(feature, fileshp=""):
    """
    SaveFeature
    """
    fileshp = fileshp if fileshp else "%d.shp" % (feature.GetField("OBJECTID"))
    driver = ogr.GetDriverByName("ESRI Shapefile")
    if os.path.exists(fileshp):
        driver.DeleteDataSource(fileshp)
    ds = driver.CreateDataSource(fileshp)
    geom = feature.GetGeometryRef()
    layer = ds.CreateLayer(fileshp, srs=geom.GetSpatialReference(), geom_type=geom.GetGeometryType())

    # create a field
    # idField = ogr.FieldDefn(fieldName, fieldType)
    # layer.CreateField(idField)

    # Create the feature and set values
    featureDefn = layer.GetLayerDefn()
    layer.CreateFeature(feature)
    feature = None
    ds = None
    return fileshp

def RasterizeLike(file_shp, file_dem, file_tif="", burn_fieldname=""):
    """
    RasterizeLike
    """
    dataset = gdal.Open(file_dem, gdalconst.GA_ReadOnly)
    vector  = ogr.OpenShared(file_shp)
    if dataset and vector:
        band = dataset.GetRasterBand(1)
        m,n = dataset.RasterYSize,dataset.RasterXSize
        gt,prj = dataset.GetGeoTransform(),dataset.GetProjection()
        nodata = band.GetNoDataValue()
        bandtype = gdal.GetDataTypeName(band.DataType)
        _, px, _, _, _, py = gt


        # Open the data source and read in the extent

        layer = vector.GetLayer()

        # Create the destination data source

        target_ds = gdal.GetDriverByName('GTiff').Create(file_tif, n, m, 1, band.DataType)
        if (gt != None):
            target_ds.SetGeoTransform(gt)
        if (prj != None):
            target_ds.SetProjection(prj)
        band = target_ds.GetRasterBand(1)
        band.SetNoDataValue(nodata)

        # Rasterize
        # gdal.RasterizeLayer(target_ds, [1], layer, burn_values=[0])
        if burn_fieldname:
            gdal.RasterizeLayer(target_ds, [1], layer, options=["ATTRIBUTE=%s" % (burn_fieldname.upper())])
        else:
            gdal.RasterizeLayer(target_ds, [1], layer, burn_values=[1])

        dataset, verctor, target_ds = None, None, None