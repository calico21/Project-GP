#include "bezier_curve.hpp"

#include <iostream>

using namespace mirena;
inline void _add_points_sorted_by_key(const std::unordered_map<double, Point3D>& map, std::vector<Point3D>& output)
{
	std::vector<double> keys;
	keys.reserve(map.size());
	for (const auto &pair : map)
	{
		keys.push_back(pair.first);
	}

	std::sort(keys.begin(), keys.end());

	for (const auto &key : keys)
	{
		output.push_back(map.at(key));
	}
}

void mirena::BezierCurve3D::insert_point(Point3D point, int insert_at)
{
	this->insert_point(point, {0, 0, 0}, {0, 0, 0}, insert_at);
}

void mirena::BezierCurve3D::insert_point(Point3D point, Point3D in_control_point, Point3D out_control_point, int insert_at)
{
	this->insert_point({point, in_control_point, out_control_point}, insert_at);
}

void mirena::BezierCurve3D::insert_point(BezierCurvePoint point, int insert_at)
{
	if(insert_at < 0){
		this->_curve_points.push_back(point);
	} else if ((size_t)insert_at < this->_curve_points.size() - 1 ) {
		this->_curve_points.insert(this->_curve_points.begin() + insert_at, point);
	}
}

BezierCurve3D::BezierCurvePoint& mirena::BezierCurve3D::get_point(int index)
{
    return _curve_points.at(index);
}

Point3D mirena::BezierCurve3D::sample(double d_t)
{
	double d_t_scaled = d_t * get_number_of_points();

	int int_part = static_cast<int>(d_t_scaled);
	double decimal_part = d_t_scaled - int_part;

	return sample_index(int_part, decimal_part);
}

Point3D mirena::BezierCurve3D::sample_index(int index, double d_t)
{
	int next_index = index + 1;
	if (_is_closed)
	{
		next_index = next_index % get_number_of_points();
	}

	auto p_start = this->_curve_points.at(index);
	auto p_end = this->_curve_points.at(next_index);

	return BezierCurve3D::_bezier_interpolate(p_start.point, p_start.point + p_start.out_control_point, p_end.point + p_end.in_control_point, p_end.point, d_t);
}

std::vector<Point3D> mirena::BezierCurve3D::tesselate(int max_subdivision_per_segment, float p_tolerance)
{
	if(_curve_points.size() == 0){
		return std::vector<Point3D>();
	}
	
	std::vector<std::unordered_map<double, Point3D>> segments;
	int amount_of_segments = _is_closed ? _curve_points.size() : _curve_points.size() - 1;

	if (amount_of_segments < 1) {
		return std::vector<Point3D>();
	}

	segments.resize(amount_of_segments);

	// Insert first point
	segments.at(0)[0] = _curve_points.at(0).point;

	for (int segment_id = 0; segment_id < amount_of_segments; segment_id ++){
		BezierCurvePoint p_a = _curve_points.at(segment_id);
		BezierCurvePoint p_b = _curve_points.at(( _is_closed && segment_id == amount_of_segments - 1 ) ? 0 : segment_id + 1);
		auto& current_segment = segments.at(segment_id);

		_bake_segment(current_segment, 0, 1, p_a.point, p_a.out_control_point, p_b.point, p_b.in_control_point,0, max_subdivision_per_segment, p_tolerance);
		current_segment[1] = p_b.point; // Add endpoint
	}

    return _flatten_segment_bake(segments);
}

std::vector<Point3D> mirena::BezierCurve3D::tesselate_even_length(int max_subdivision_per_segment, float p_length)
{
    	int amount_of_segments = _is_closed ? _curve_points.size() : _curve_points.size() - 1;
	std::vector<std::unordered_map<double, Point3D>> segments;

	if (amount_of_segments < 1) {
		return std::vector<Point3D>();
	}

	segments.resize(amount_of_segments);

	// Insert first point
	segments.at(0)[0] = _curve_points.at(0).point;

	for (int segment_id = 0; segment_id < amount_of_segments; segment_id ++){
		BezierCurvePoint p_a = _curve_points.at(segment_id);
		BezierCurvePoint p_b = _curve_points.at(( _is_closed && segment_id == amount_of_segments - 1 ) ? 0 : segment_id + 1);
		auto& current_segment = segments.at(segment_id); //antes auto

		_bake_segment_even_length(current_segment, 0, 1, p_a.point, p_a.out_control_point, p_b.point, p_b.in_control_point, 0, max_subdivision_per_segment, p_length);
		current_segment[1] = p_b.point;

	}

    return _flatten_segment_bake(segments);
}

inline std::vector<Point3D> mirena::BezierCurve3D::_flatten_segment_bake(std::vector<std::unordered_map<double, Point3D>> segments)
{
	std::vector<Point3D> ret_points;

	int point_count = 0;
	for (std::unordered_map<double, Point3D> &E : segments)
	{
		point_count = point_count + E.size(); 
	}

	ret_points.reserve(point_count);

	for (size_t segment_id = 0; segment_id < segments.size() ; segment_id++)
	{
		_add_points_sorted_by_key(segments.at(segment_id), ret_points); /* Add midpoints */
	}

	return ret_points;
}

void mirena::BezierCurve3D::_bake_segment(std::unordered_map<double, Point3D> &r_bake, double p_begin, double p_end, const Point3D &p_a, const Point3D &p_out, const Point3D &p_b, const Point3D &p_in, int p_depth, int p_max_depth, double p_tol) const
{
	double mp = p_begin + (p_end - p_begin) * 0.5;
	Point3D beg = _bezier_interpolate(p_a, p_a + p_out, p_b + p_in, p_b, p_begin);
	Point3D mid = _bezier_interpolate(p_a, p_a + p_out, p_b + p_in, p_b, mp);
	Point3D end = _bezier_interpolate(p_a, p_a + p_out, p_b + p_in, p_b, p_end);

	Point3D na = (mid - beg).normalized();
	Point3D nb = (end - mid).normalized();
	double dp = na.dot(nb);

	if (dp > cos(p_tol))
	{
		r_bake[mp] = mid;
	}
	if (p_depth < p_max_depth)
	{
		_bake_segment(r_bake, p_begin, mp, p_a, p_out, p_b, p_in, p_depth + 1, p_max_depth, p_tol);
		_bake_segment(r_bake, mp, p_end, p_a, p_out, p_b, p_in, p_depth + 1, p_max_depth, p_tol);
	}
}

void mirena::BezierCurve3D::_bake_segment_even_length(std::unordered_map<double, Point3D> &r_bake, double p_begin, double p_end, const Point3D &p_a, const Point3D &p_out, const Point3D &p_b, const Point3D &p_in, int p_depth, int p_max_depth, double p_length) const
{
	Point3D beg = _bezier_interpolate(p_a, p_a + p_out, p_b + p_in, p_b, p_begin);
	Point3D end = _bezier_interpolate(p_a, p_a + p_out, p_b + p_in, p_b, p_end);

	double length = beg.distance_to(end);

	if (length > p_length && p_depth < p_max_depth)
	{
		double mp = (p_begin + p_end) * 0.5;
		Point3D mid = _bezier_interpolate(p_a, p_a + p_out, p_b + p_in, p_b, mp);
		r_bake[mp] = mid;

		_bake_segment_even_length(r_bake, p_begin, mp, p_a, p_out, p_b, p_in, p_depth + 1, p_max_depth, p_length);
		_bake_segment_even_length(r_bake, mp, p_end, p_a, p_out, p_b, p_in, p_depth + 1, p_max_depth, p_length);
	}
}