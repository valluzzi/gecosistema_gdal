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
import os,sys
import math,json
import ogr,osr,rtree
import gdal,gdalconst
from gecosistema_core import *
from gdal2numpy import GDAL2Numpy,Numpy2GTiff


def CreateSpatialIndex(fileshp):
    """
    CreateSpatialIndex
    """
    fileidx = forceext(fileshp,"idx")
    dataset = ogr.OpenShared(fileshp)
    if dataset:
        indexname = forceext(fileidx,"")
        if not os.path.isfile(fileidx):
            index = rtree.index.Index(indexname)
            layer = dataset.GetLayer(0)
            layer.ResetReading()
            for feature in layer:
                if feature.GetGeometryRef():
                    minx,miny,maxx,maxy = feature.GetGeometryRef().GetEnvelope()
                    index.insert(feature.GetFID(), (minx,maxx,miny,maxy))
        else:
            try:
                index = rtree.index.Index(indexname)
            except Exception as ex:
                print("CreateSpatialIndex:",ex)
                index=None
        return index
    return None

def GetSpatialIndex(fileshp):
    """
    GetSpatialIndex
    """
    index = None
    if not os.path.isfile(forceext(fileshp,"idx")):
        return None
    try:
        index = rtree.index.Index(forceext(fileshp, ""))
    except Exception as ex:
        print("CreateSpatialIndex:", ex)
    return index

def UpdateSpatialIndex(fileshp, features):
    """
    UpdateSpatialIndex
    """
    index = None
    if os.path.isfile(forceext(fileshp, "idx")):
        try:
            index = rtree.index.Index(forceext(fileshp, ""))
            for feature in listify(features):
                minx, miny, maxx, maxy = feature.GetGeometryRef().GetEnvelope()
                index.insert(feature.GetFID(), (minx, maxx, miny, maxy))
        except Exception as ex:
            print("UpdateSpatialIndex:", ex)
        return index


def ExportToJson(feature, fieldnames=[], coord_precision=2):
    """
    ExportToJson
    """
    n = len(fieldnames)
    geom = feature.GetGeometryRef() #.Simplify(20.0)
    geometry_type = geom.GetGeometryName().capitalize()

    if geometry_type =="Point":
        x,y = geom.GetPoints()[0]
        coords = [round(x,coord_precision), round(y,coord_precision)]
    elif geometry_type =="Linestring":
        geometry_type = "LineString"
        coords = [ list(p) for p in geom.GetPoints() ]
        coords = [ [round(x,coord_precision), round(y,coord_precision)]  for x,y in coords ]
    elif geometry_type =="Multiline":
        geometry_type = "MultiLine"
        coords = [[[[round(x,coord_precision), round(y,coord_precision)] for x,y in ring.GetPoints()] for ring in line if ring.GetPointCount()] for line in geom]
    elif geometry_type =="Polygon":
        coords = [[[round(x,coord_precision), round(y,coord_precision)] for x,y in ring.GetPoints()] for ring in geom if ring.GetPointCount()]
    elif geometry_type =="Multipolygon":
        geometry_type = "MultiPolygon"
        coords = [[[[round(x,coord_precision), round(y,coord_precision)] for x,y in ring.GetPoints()] for ring in poly if ring.GetPointCount()] for poly in geom]
    else:
        print("TODO:",geometry_type)

    if len(coords)==0:
        return False

    props  = {}
    for j in range(n):
        props[fieldnames[j]]=feature.GetField(fieldnames[j])

    return {
        "id": feature.GetFID(),
        "type": "Feature",
        "geometry": {
            "type":geometry_type,
            "coordinates": coords
        },
        "properties": props
    }

def GetSpatialRef(fileshp):
    """
    GetSpatialRef
    """
    srs = None
    if isinstance(fileshp,(str,)) and os.path.isfile(fileshp):
        dataset = ogr.OpenShared(fileshp)
        if dataset:
            layer = dataset.GetLayer()
            srs = layer.GetSpatialRef()
        dataset = None

    elif isinstance(fileshp,(str,))  and "epsg:" in fileshp.lower():
        code = int(fileshp.lower().replace("epsg:",""))
        srs = osr.SpatialReference()
        srs.ImportFromEPSG(code)

    else:
        code = int(fileshp)
        if code>0:
            srs = osr.SpatialReference()
            srs.ImportFromEPSG(code)

    return srs

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


def GetFeatureBy(fileshp, layername=0, attrname="ogr_id", attrvalue=0 ):
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
                    #patch geometry that sometime is invalid
                    #create a buffer of 0 meters
                    buff0m = feature.GetGeometryRef().Buffer(0)
                    feature.SetGeometry(buff0m)
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

def queryByPoint( fileshp, x=0, y=0, point_srs=None, mode="single"):
    """
    queryByPoint
    """
    res = []
    point = ogr.Geometry(ogr.wkbPoint)
    point.AddPoint(x,y)

    dataset = ogr.OpenShared(fileshp)
    if dataset:
        layer = dataset.GetLayer(0)
        srs = layer.GetSpatialRef()
        if point_srs:
            psrs = osr.SpatialReference()
            psrs.ImportFromEPSG(int(point_srs))
            if  not psrs.IsSame(srs):
                transform = osr.CoordinateTransformation(psrs, srs)
                point.Transform(transform)

        for feature in layer:
            geom = feature.GetGeometryRef()
            if point.Intersects( geom ):
                res.append(feature)
                if mode.lower()=="single":
                    break
    dataset = None
    return res

def queryByAttributes( fileshp, fieldname, fieldvalues, mode="multiple"):
    """
    queryByAttributes
    """
    res = []
    dataset = ogr.OpenShared(fileshp)
    if dataset:
        layer = dataset.GetLayer(0)
        for feature in layer:
            if feature.GetFieldIndex(fieldname)>=0:
                if feature.GetField(fieldname) in listify(fieldvalues):
                    res.append(feature)
                    if mode.lower()=="single":
                        break
    dataset = None
    return res

def queryByShape( fileshp, feature, feature_epsg=None, mode="single"):
    """
    queryByShape
    """
    res = []
    if not feature:
        return []

    dataset = ogr.OpenShared(fileshp)
    if dataset:
        layer = dataset.GetLayer(0)
        srs = layer.GetSpatialRef()

        qshape = feature.GetGeometryRef() if isinstance(feature, ogr.Feature) else feature

        if feature_epsg:
            qsrs = osr.SpatialReference()
            qsrs.ImportFromEPSG(int(feature_epsg))
            if  not qsrs.IsSame(srs):
                #transform the query feature in layer srs
                transform = osr.CoordinateTransformation(qsrs, srs)
                qshape.Transform(transform)

        """
        # 2) 1st sequential approach
        for feature in layer:
            geom = feature.GetGeometryRef()
            if qshape.Intersects( geom ):
                res.append(feature)
                if mode.lower()=="single":
                    break
        """

        # 2) Rtree index approach
        fileidx = forceext(fileshp,".idx")
        if os.path.isfile(fileidx):
            minx,miny,maxx,maxy = qshape.GetEnvelope()
            for fid in list(index.intersection((minx, maxx, miny, maxy))):
                feature = layer.GetFeature(fid)
                if feature:
                    res.append(feature)
                    if mode.lower() == "single":
                        break
        else:
            # 3) Spatial filter approach
            geom = feature.GetGeometryRef()
            layer.SetSpatialFilter(qshape)
            layer.ResetReading()
            for feature in layer:
                res.append(feature)
                if mode.lower() == "single":
                    break


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

def CreateShapefile(fileshp, crs=4326, schema={}):
    """
    CreateShapefile

    schema={"geometry":"LineString","properties":{"OBJECTID":"int","height":"float"}}
    """
    DATATYPE={
        "Point":ogr.wkbPoint,
        "LineString":ogr.wkbLineString,
        "Polygon":ogr.wkbPolygon,
        "MultiPoint":ogr.wkbMultiPoint,
        "MultiLineString":ogr.wkbMultiLineString,
        "MultiPolygon":ogr.wkbMultiPolygon,
        "int": ogr.OFTInteger,
        "float":ogr.OFTReal,
        "str":ogr.OFTString,
        "text":ogr.OFTString
    }

    layername = juststem(fileshp)
    # set up the shapefile driver
    driver = ogr.GetDriverByName("ESRI Shapefile")

    # create the data source
    data_source = driver.CreateDataSource(fileshp)

    # create the spatial reference, WGS84
    srs = GetSpatialRef(crs)

    # create the layer
    #layer = data_source.CreateLayer(layername, srs, ogr.wkbPoint)
    dtype = schema["geometry"] if "geometry" in schema else "Point"
    layer = data_source.CreateLayer(layername, srs, DATATYPE[dtype])

    properties = schema["properties"] if "properties" in schema else {}
    for name in properties:
        dtype,dwidth = (properties[name]+":0").split(":")
        p,w =  math.modf(float(dwidth))
        p,w = int(p),int(w)
        field_name = ogr.FieldDefn(name, DATATYPE[dtype])
        p =255 if p==0 and dtype=="str" else p
        if w:
            field_name.SetWidth(w)
        if p:
            field_name.SetPrecision(p)
        layer.CreateField(field_name)
    # Save and close the data source
    data_source = None

def AddField(fileshp, fieldname, fieldtype, fieldsize="12.4", fieldvalue=None ):
    """
    AddField
    """
    ds = ogr.Open(fileshp,1)
    if ds:
        DATATYPE = {
            "Point": ogr.wkbPoint,
            "LineString": ogr.wkbLineString,
            "Polygon": ogr.wkbPolygon,
            "MultiPoint": ogr.wkbMultiPoint,
            "MultiLineString": ogr.wkbMultiLineString,
            "MultiPolygon": ogr.wkbMultiPolygon,
            "int": ogr.OFTInteger,
            "float": ogr.OFTReal,
            "str": ogr.OFTString,
            "text": ogr.OFTString
        }
        layer = ds.GetLayer()
        w, p  = (fieldsize+ ".0").split(".")[0:2]
        p, w = int(p), int(w)
        fielddefn = ogr.FieldDefn(fieldname, DATATYPE[fieldtype])
        w = 255 if w == 0 and fieldtype in ("str","text") else p
        if w:
            fielddefn.SetWidth(w)
        if p:
            fielddefn.SetPrecision(p)
        if fieldvalue!=None:
            fielddefn.SetDefault(fieldvalue)
        layer.CreateField(fielddefn)

        #update features to fdefault value
        if fieldvalue!=None:
            layer.ResetReading()
            for feature in layer:
                feature.SetField(fieldname, fieldvalue)
                layer.SetFeature(feature)

        layer, ds = None,None


def GetFeatureByAttribute(layer, attrname="OBJECTID", attrvalue=0):
    """
    GetFeatureByAttribute
    """
    if layer:
        layerDefinition = layer.GetLayerDefn()
        fieldnames = [layerDefinition.GetFieldDefn(j).GetName().upper() for j in range(layerDefinition.GetFieldCount())]
        if attrname.upper() in fieldnames:
            for feature in layer:
                if feature.GetField(attrname) == attrvalue:
                    dataset = None
                    # patch geometry that sometime is invalid
                    # create a buffer of 0 meters
                    buff0m = feature.GetGeometryRef().Buffer(0)
                    feature.SetGeometry(buff0m)
                    return feature
    return None


def WriteRecords(fileshp, records, src_epsg=-1):
    """
    WriteRecord
    """
    mode = "insert"
    datasource = ogr.Open(fileshp,1)
    if datasource:
        layer = datasource.GetLayer()
        dsr   = layer.GetSpatialRef()
        srs   = GetSpatialRef(src_epsg)

        layerDefinition = layer.GetLayerDefn()
        fieldnames = [layerDefinition.GetFieldDefn(j).GetName() for j in range(layerDefinition.GetFieldCount())]

        for record in records:
            properties = record["properties"] if "properties" in record else {}
            #fid = int(properties["FID"]) if "FID" in properties else -1
            #Case wms (gml)
            if "id" in record and isstring(record["id"]) and "." in record["id"]:
                fid = int(record["id"].split(".")[1])
            elif "id" in record:
                fid = int(record["id"])
            else:
                fid = -1

            # create the feature
            mode= "update"
            feature = layer.GetFeature(fid)
            if not feature:
                mode = "insert"
                feature = ogr.Feature(layerDefinition)

            # Set the attributes using the values from the delimited text file
            if properties:
                for name in properties:
                    if not name in ("boundedBy",):
                        value = properties[name]
                        #print("SetField(%s,%s)"%(name,value))
                        feature.SetField(name, value)

            # create the WKT for the feature using Python string formatting
            if "geometry" in record:
                geojson = json.dumps(record["geometry"])
                geom = ogr.CreateGeometryFromJson(geojson)
                #srs  = geom.GetSpatialReference() #usually dont work or noinfo

                if srs and not dsr.IsSame(srs):
                    transform = osr.CoordinateTransformation(srs, dsr)
                    geom.Transform(transform)

                feature.SetGeometry(geom)


            if mode=="insert":
                layer.CreateFeature(feature)
                fid = feature.GetFID()
                for fieldname in ("FID","OBJECTID",):
                    if fieldname in fieldnames:
                        feature.SetField(fieldname, fid)
                layer.SetFeature(feature)
                # update the spatial index
                UpdateSpatialIndex(fileshp, feature)

            else:
                layer.SetFeature(feature)
                # update the spatial index
                UpdateSpatialIndex(fileshp, feature)
            feature = None

    # Save and close the data source
    datasource = None

def DeleteRecords(fileshp, fids=None):
    """
    DeleteRecords
    """
    datasource = ogr.Open(fileshp, 1)
    if datasource:
        layer = datasource.GetLayer()
        if fids:
            for fid in listify(fids):
                layer.DeleteFeature(fid)
        else:
            for feature in layer:
                layer.DeleteFeature(feature.GetFID())
    datasource = None

def UpdateRecordsByAttribute(fileshp, attrnames, values):
    """
    UpdateRecordsByAttribute(s)
    """
    datasource = ogr.Open(fileshp, 1)
    if datasource:
        layer = datasource.GetLayer()
        layerDefinition = layer.GetLayerDefn()
        fieldnames = [layerDefinition.GetFieldDefn(j).GetName() for j in range(layerDefinition.GetFieldCount())]
        attrnames  = listify(attrnames)
        values     = listify(values)
        n = min(len(attrnames),len(values))
        for feature in layer:
            something_has_changed = False
            for j in range(n):
                attrname = attrnames[j]
                if attrname in fieldnames:
                    value = values[j]
                    if isinstance(value,(str,)) and value in fieldnames:
                        value = feature.GetField(value)
                    feature.SetField(attrname,value)
                    something_has_changed = True
            if something_has_changed:
                layer.SetFeature(feature)
    datasource = None


def DeleteRecordsByAttribute(fileshp, attrname, values):
    """
    DeleteRecordsByAttribute
    """
    datasource = ogr.Open(fileshp, 1)
    if datasource:
        layer = datasource.GetLayer()
        layerDefinition = layer.GetLayerDefn()
        fieldnames = [layerDefinition.GetFieldDefn(j).GetName() for j in range(layerDefinition.GetFieldCount())]
        if attrname in fieldnames:
            for feature in layer:
                for value in listify(values):
                    if feature.GetField(attrname) == value:
                        layer.DeleteFeature(feature.GetFID())
                        break
    datasource = None

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








