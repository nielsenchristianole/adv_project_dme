import shapely


def apply_modulo_to_geom(
    geom: shapely.MultiPolygon | shapely.Polygon,
    *,
    mod_x: float | None = None,
    mod_y: float | None = None
) -> shapely.MultiPolygon | shapely.Polygon:

    def apply_modulo_to_coords(coords):
        return [
            (
                x % mod_x if mod_x is not None else x,
                y % mod_y if mod_y is not None else y
            ) for x, y in coords]

    def apply_modulo_to_polygon(polygon: shapely.Polygon):
        new_exterior = apply_modulo_to_coords(polygon.exterior.coords)
        new_interiors = [apply_modulo_to_coords(interior.coords) for interior in polygon.interiors]
        return shapely.Polygon(new_exterior, new_interiors)
    
    if isinstance(geom, shapely.Polygon):
        return apply_modulo_to_polygon(geom)
    else:
        assert isinstance(geom, shapely.MultiPolygon)

    return shapely.MultiPolygon([apply_modulo_to_polygon(polygon) for polygon in geom.geoms])


def project_coords_to_axis(
    geom: shapely.MultiPolygon | shapely.Polygon,
    *,
    x_axis: bool = True
) -> list[float]:

    def project_coords(coords):
        return [x if x_axis else y for x, y in coords]

    def project_polygon(polygon: shapely.Polygon):
        points = project_coords(polygon.exterior.coords)
        for interior in polygon.interiors:
            points.extend(project_coords(interior.coords))
        return points
    
    if isinstance(geom, shapely.Polygon):
        return project_polygon(geom)
    else:
        assert isinstance(geom, shapely.MultiPolygon)

    out_points = list()
    for polygon in geom.geoms:
        out_points.extend(project_polygon(polygon))
    return out_points
