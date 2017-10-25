import ee
import time
import sys
import numpy as np
import pandas as pd
import itertools
import os
import urllib
import json

ee.Initialize()

def export_oneimage(img,folder,name,region,scale,crs):
  task = ee.batch.Export.image(img, name, {
      'driveFolder':folder,
      'driveFileNamePrefix':name,
      'region': region,
      'scale':scale,
      'crs':crs
  })
  task.start()
  while task.status()['state'] in ['READY', 'RUNNING']:
    print 'Running...'
    print task.status()
    # Perhaps task.cancel() at some point.
    time.sleep(10)
  print 'Done.', task.status()


data = pd.read_csv('data_for_image_download.csv',header=None)

def appendBand(current, previous):
    previous=ee.Image(previous)
    current = current.select([0,1,2,3,4,5,6])
    accum = ee.Algorithms.If(ee.Algorithms.IsEqual(previous,None), current, previous.addBands(ee.Image(current)))
    return accum

#world_region = ee.FeatureCollection('ft:1tdSwUL7MVpOauSgRzqVTOwdfy17KDbw-1d9omPw')

imgcoll = ee.ImageCollection('MODIS/MOD09A1').filterDate('2016-06-01','2016-12-01')#.sort('system:time_start', True)
img=imgcoll.iterate(appendBand)
img=ee.Image(img)

for hhid, yld, lat, lon in data.values:
    fname = '{}-{}'.format(str(hhid).replace(".", "_"), str(yld).replace(".","_"))
    offset = 0.005 #TODO Change this
    scale  = 500
    crs='EPSG:4326'

    region = str(ee.Geometry.Rectangle(lon - offset, lat - offset, lon + offset, lat + offset)['coordinates'])

    print region

    export_oneimage(img,'Data',fname,region,scale,crs)
