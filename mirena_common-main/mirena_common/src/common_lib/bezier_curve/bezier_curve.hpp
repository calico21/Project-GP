#pragma once

#include <vector>
#include <unordered_map>
#include <cmath>
#include <algorithm>

// Most of this is based off the godot engine source code, with a few changes, so check it out if in doubt (godot/scenes/resources/curve, version 4.4)

namespace mirena
{
    struct Point3D
    {
        double x;
        double y;
        double z;

        inline Point3D operator+(const Point3D &other) const
        {
            Point3D ret;
            ret.x = this->x + other.x;
            ret.y = this->y + other.y;
            ret.z = this->z + other.z;
            return ret;
        }

        inline Point3D operator-(const Point3D &other) const
        {
            Point3D ret;
            ret.x = this->x - other.x;
            ret.y = this->y - other.y;
            ret.z = this->z - other.z;
            return ret;
        }

        inline Point3D operator*(double scalar) const
        {
            Point3D ret;
            ret.x = this->x * scalar;
            ret.y = this->y * scalar;
            ret.z = this->z * scalar;
            return ret;
        }

        inline Point3D operator/(double scalar) const
        {
            Point3D ret;
            ret.x = this->x / scalar;
            ret.y = this->y / scalar;
            ret.z = this->z / scalar;
            return ret;
        }

        inline double dot(const Point3D &other) const
        {
            return this->x * other.x + this->y * other.y + this->z * other.z;
        }

        inline double mod() const
        {
            return sqrt(this->dot(*this));
        }

        inline Point3D normalized() const
        {
            return *this / this->mod();
        }

        inline Point3D rotated_by_yaw(double yaw) const
        {
            const double cos_yaw = std::cos(yaw);
            const double sin_yaw = std::sin(yaw);
            return {
                this->x * cos_yaw - this->y * sin_yaw, 
                this->x * sin_yaw + this->y * cos_yaw, 
                this->z
            };
        }

        inline double distance_to(Point3D &other)
        {
            return (*this - other).mod();
        }
    };

    class BezierCurve3D
    {

    public:
        struct BezierCurvePoint
        {
            Point3D point;
            Point3D in_control_point;
            Point3D out_control_point;
        };

        int get_number_of_points() { return this->_curve_points.size(); }

        bool is_closed() { return this->_is_closed; }
        void set_is_closed(bool value) { this->_is_closed = value; }

        // Insert a new bezier curve point
        // If 'insert_at' is -1, its inserted at the end of the curve. Else, its inserted before the point specified
        void insert_point(Point3D point, int insert_at = -1);
        void insert_point(Point3D point, Point3D in_control_point, Point3D out_control_point, int insert_at = -1);
        void insert_point(BezierCurvePoint point, int insert_at = -1);
        BezierCurvePoint &get_point(int index);

        // Reserve memory for the speficied number of points the internal BezierCurvePoint vector to speed up bulk insertions
        void reserve(size_t number_of_points) { this->_curve_points.reserve(number_of_points); };

        // Sample over the whole curve; d_t must be contained between 0 and 1
        // Does not ensure a smooth speed over the curve.
        Point3D sample(double d_t);

        // Sample between 2 curve points.  d_t must be contained between 0 and 1
        Point3D sample_index(int index, double d_t);

        // Samples the segment of the curve. The density of the sampling depends on the curviness of the curve
        // p_tolerance is the maximum angle formed between 3 consecutive points in radians (defaults to 6 degrees)
        std::vector<Point3D> tesselate(int max_subdivision_per_segment = 6, float p_tolerance = M_PI / 30);

        // Samples the segment of the curve. The density of the sampling is constant
        // p_length determines the maximun distance between 2 consecutive points
        std::vector<Point3D> tesselate_even_length(int max_subdivision_per_segment = 6, float p_length = 0.2);

    private:
        std::vector<BezierCurvePoint> _curve_points;
        bool _is_closed;

        inline std::vector<Point3D> _flatten_segment_bake(std::vector<std::unordered_map<double, Point3D>> midpoints);

        // Samples the segment of the curve. The density of the sampling depends on the curviness of the curve
        // p_tol is the maximum angle formed between 3 consecutive points in radians
        void _bake_segment(std::unordered_map<double, Point3D> &r_bake, double p_begin, double p_end, const Point3D &p_a, const Point3D &p_out, const Point3D &p_b, const Point3D &p_in, int p_depth, int p_max_depth, double p_tol) const;

        // Samples the segment of the curve. The density of the sampling is constant
        // p_lenght determines the maximun distance between 2 consecutive points
        void _bake_segment_even_length(std::unordered_map<double, Point3D> &r_bake, double p_begin, double p_end, const Point3D &p_a, const Point3D &p_out, const Point3D &p_b, const Point3D &p_in, int p_depth, int p_max_depth, double p_length) const;

        inline static double _bezier_interpolate(double p_start, double p_control_1, double p_control_2, double p_end, double p_t)
        {
            /* Formula from Wikipedia article on Bezier curves. */
            double omt = (1.0f - p_t);
            double omt2 = omt * omt;
            double omt3 = omt2 * omt;
            double t2 = p_t * p_t;
            double t3 = t2 * p_t;

            return p_start * omt3 + p_control_1 * omt2 * p_t * 3.0f + p_control_2 * omt * t2 * 3.0f + p_end * t3;
        }

        inline static Point3D _bezier_interpolate(Point3D p_start, Point3D p_control_1, Point3D p_control_2, Point3D p_end, double p_t)
        {
            /* Formula from Wikipedia article on Bezier curves. */
            Point3D ret;
            ret.x = _bezier_interpolate(p_start.x, p_control_1.x, p_control_2.x, p_end.x, p_t);
            ret.y = _bezier_interpolate(p_start.y, p_control_1.y, p_control_2.y, p_end.y, p_t);
            ret.z = _bezier_interpolate(p_start.z, p_control_1.z, p_control_2.z, p_end.z, p_t);
            return ret;
        }
    };
}