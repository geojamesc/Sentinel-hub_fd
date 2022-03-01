import click
import fiona
from fiona.crs import from_epsg
from shapely.geometry import mapping
from shapely.wkt import loads
import rasterio
import csv
import os


def get_wkt(minx, miny, maxx, maxy):

    wkt = """POLYGON(({0} {1},{2} {3},{4} {5},{6} {7},{0} {1}))""".format(
        minx,
        miny,
        maxx,
        miny,
        maxx,
        maxy,
        minx,
        maxy
    )

    return wkt


def write_csv_to_shp():

    my_schema = {
        "geometry": "Polygon",
        "properties": {
            "id": "int",
            "patchlet": "str"
        }
    }

    my_driver = "ESRI Shapefile"
    my_crs = from_epsg(32632)
    out_fname = '/home/james/Desktop/patchlets_bboxes.shp'

    with fiona.open(out_fname, "w", driver=my_driver, crs=my_crs, schema=my_schema) as my_collection:
        with open('/home/james/Desktop/patchlets_bboxes.csv', 'r') as inpf:
            my_reader = csv.DictReader(inpf)
            id = 1
            for r in my_reader:
                pth_to_patchlet = r['pth_to_patchlet']
                patchlet = os.path.split(pth_to_patchlet)[-1]
                min_x, min_y, max_x, max_y = r['min_x'], r['min_y'], r['max_x'], r['max_y']
                wkt = get_wkt(min_x, min_y, max_x, max_y)
                print(id, patchlet, wkt)
                id += 1

                my_collection.write({
                    "geometry": mapping(loads(wkt)),
                    "properties": {
                        "id": id,
                        "patchlet": patchlet
                    }
                })


def main():
    write_csv_to_shp()


if __name__ == "__main__":
    main()