import os
import subprocess
import shutil

# assumes that have under /home/james/Work/FieldBoundaries/Simon_Data/070222_reprojected_to_wgs84/Sentinel-hub_fd/input-data/tiffs
# have these empty subfolders created:
# 32VNH_1
# 32VNH_2
# 32VNH_3

# [1] reproject tiffs to EPSG:4326 (WGS84) and copy date.txt metadata file

print('[1] Reprojecting tiff bands to EPSG:4326')
base_path = '/home/james/Work/FieldBoundaries/Simon_Data/070222/Sentinel-hub_fd/input-data/tiffs'
new_path = '/home/james/Work/FieldBoundaries/Simon_Data/070222_reprojected_to_wgs84/Sentinel-hub_fd/input-data/tiffs'
for root, folders, files in os.walk(base_path):
    for fn in files:
        # run gdalwarp to reproject all tiff data to WGS84
        # specifying -cutline we could clip?
        if os.path.splitext(fn)[-1] == '.tif':
            src_tif = os.path.join(root, fn)
            dst_tif = os.path.join(new_path, os.path.split(os.path.split(src_tif)[0])[1], fn)
            gdal_cmd = 'gdalwarp -t_srs EPSG:4326 {0} {1}'.format(
                src_tif,
                dst_tif
            )
            print(gdal_cmd)
            subprocess.call(gdal_cmd, shell=True)
        # copy the date.txt file which accompanies each set of tiffs
        if os.path.splitext(fn)[-1] == '.txt':
            if fn == 'date.txt':
                src_txt = os.path.join(root, fn)
                dst_txt = os.path.join(new_path, os.path.split(os.path.split(src_txt)[0])[1], fn)
                shutil.copyfile(src_txt, dst_txt)

# [2] reproject geopackages to EPSG:4326 (WGS84) and copy province border which should already be in WGS84
print('[2] Reprojecting grid definition and fields geopackages to EPSG:4326')
src_geopackages = [
    '/home/james/Work/FieldBoundaries/Simon_Data/070222/Sentinel-hub_fd/input-data/cyl-grid-definition.gpkg',
    '/home/james/Work/FieldBoundaries/Simon_Data/070222/Sentinel-hub_fd/input-data/fields.gpkg'
]

for src_fn in src_geopackages:
    if os.path.exists(src_fn):
        dst_fn = src_fn.replace('070222', '070222_reprojected_to_wgs84')
        ogr_cmd = 'ogr2ogr -f GPKG -t_srs EPSG:4326 {0} {1}'.format(
            dst_fn, src_fn
        )
        print(ogr_cmd)
        subprocess.call(ogr_cmd, shell=True)

# [3] reproject province-border.geojson to EPSG:4326
print('[3] Reproject province-border.geojson')

src_province_border_geojson_fn = '/home/james/Work/FieldBoundaries/Simon_Data/070222/Sentinel-hub_fd/input-data/cyl-province-border.cyl-province-border.geojson'
dst_province_border_geojson_fn = src_province_border_geojson_fn.replace('070222', '070222_reprojected_to_wgs84')

ogr_cmd = 'ogr2ogr -f GeoJSON -t_srs EPSG:4326 {0} {1}'.format(
    dst_province_border_geojson_fn, src_province_border_geojson_fn
)
print(ogr_cmd)
subprocess.call(ogr_cmd, shell=True)

# [4] run some checks afterwards on all reprojected tiffs under the 32VNH_1; 32VNH_2; 32VNH_3 sub-folders
# so under e.g. /home/james/Work/FieldBoundaries/Simon_Data/070222_reprojected_to_wgs84/Sentinel-hub_fd/input-data/tiffs/32VNH_1
# run: for i in *.tif; do echo $i; gdalinfo $i | grep ^Size; done
# --> Size is 2196, 1994
# run: for i in *.tif; do echo $i; gdalinfo $i | grep ^'Data axis'; done
# --> Data axis to CRS axis mapping: 2,1
# run: for i in *.tif; do echo $i; gdalinfo $i | grep ' ID'; done
# --> ID["EPSG",4326]]

# [5] run some checks on the reprojected vectors
# for i in *.*; do echo $i; ogrinfo -so -al $i | grep 'Data axis'; done
# --> Data axis to CRS axis mapping: 2,1
# for i in *.*; do echo $i; ogrinfo -so -al $i | grep ' ID'; done
# --> ID["EPSG",4326]]

# [6] eyeball in QGIS...










