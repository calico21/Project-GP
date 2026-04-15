#include "dbmmk1.hpp"

#include <math.h>
#include <iostream>

using namespace DBMMK1;

mirena::ExtendedKalmanFilter<STATE_DIM, CONTROL_DIM, MEASURE_DIM> DBMMK1::Model::build_ekf(const Parameters &model_parameters)
{
    // Use this model as both a Measure and Control model
    std::shared_ptr<Model> model = std::make_shared<Model>(model_parameters);
    return mirena::ExtendedKalmanFilter<STATE_DIM, CONTROL_DIM, MEASURE_DIM>(model, model);
}

X DBMMK1::Model::predict_state(const X &previous_state, const U &control)
{
    // Preparation, and pre-calculation of Fy1 & Fy2
    X prediction;
    StateAcessor x(previous_state);
    ControlAcessor u(control);

    // At u = 0 the sideslip is both negligible and undefined. For this reason, for u < u_min, we make these forces 0. [u_min, u_max] is a transition period to avoid a sudden change in Fy. We interpolate
    const double u_min = 1.5;
    const double u_max = 2.5;
    double Fy1;
    double Fy2;

    if (std::abs(x.u()) <= u_min)
    {
        Fy1 = 0;
        Fy2 = 0;
    }
    else
    {
        double interpolation_scale = std::min(1.0, (std::abs(x.u()) - u_min) / (u_max - u_min));
        Fy1 = interpolation_scale * _params.k_f * (((x.v() + _params.l_f * x.omega()) / (x.u())) - u.delta());
        Fy2 = interpolation_scale * _params.k_r * (((x.v() - _params.l_r * x.omega()) / (x.u())));

        std::cout << "FY1: " << Fy1 << ", FY2: " << Fy2 << ", scale: " << interpolation_scale << std::endl;
    }

    // Obtention of the actual prediction
    /* p_x   */ prediction(0) = x.p_x()   + u.dt() * (x.u() * cos(x.phi()) - x.v() * sin(x.phi()));
    /* p_y   */ prediction(1) = x.p_y()   + u.dt() * (x.u() * sin(x.phi()) + x.v() * cos(x.phi()));
    /* phi   */ prediction(2) = x.phi()   + u.dt() * (x.omega());
    /* u     */ prediction(3) = x.u()     + u.dt() * (u.a() + x.v() * x.omega() - (Fy1 * sin(u.delta()) / _params.m));
    /* v     */ prediction(4) = x.v()     + u.dt() * (-x.u() * x.omega() + (Fy1 * cos(u.delta()) + Fy2) / _params.m);
    /* omega */ prediction(5) = x.omega() + u.dt() * (_params.l_f * Fy1 * cos(u.delta()) - _params.l_r * Fy2) / _params.I_z;

    return prediction;
}

EMatrix<double, STATE_DIM, STATE_DIM> DBMMK1::Model::get_state_jacobian(const X &state, const U &control)
{
    // Preparation
    EMatrix<double, STATE_DIM, STATE_DIM> J = EMatrix<double, STATE_DIM, STATE_DIM>::Zero();
    const EMatrix<double, STATE_DIM, STATE_DIM> I = EMatrix<double, STATE_DIM, STATE_DIM>::Identity();
    StateAcessor x(state);
    ControlAcessor u(control);

    // Pre-Calculation of all the relevant partial derivatives of Fy1 and Fy2
    double Fy1_du = -_params.k_f * (x.v() + _params.l_f * x.omega()) / (x.u() * x.u());
    double Fy1_dv = _params.k_f / (x.u());
    double Fy1_dw = _params.k_f * (_params.l_f) / (x.u());
    double Fy2_du = -_params.k_r * (x.v() + _params.l_r * x.omega()) / (x.u() * x.u());
    double Fy2_dv = _params.k_r / (x.u());
    double Fy2_dw = _params.k_r * (_params.l_r) / (x.u());

    // Calculate the jacobian
    // Row 0
    J(0, 2) = -x.u() * sin(x.phi()) - x.v() * cos(x.phi());
    J(0, 3) = cos(x.phi());
    J(0, 4) = -sin(x.phi());

    // Row 1
    J(1, 2) = x.u() * cos(x.phi()) - x.v() * sin(x.phi());
    J(1, 3) = sin(x.phi());
    J(1, 4) = cos(x.phi());

    // Row 2
    J(2, 5) = 1.0;

    // Row 3:
    J(3, 3) = -(Fy1_du * sin(u.delta())) / _params.m;
    J(3, 4) = x.omega() - (Fy1_dv * sin(u.delta())) / _params.m;
    J(3, 5) = x.v() - (Fy1_dw * sin(u.delta())) / _params.m;

    // Row 4:
    J(4, 3) = -x.omega() + (Fy1_du * cos(u.delta()) + Fy2_du) / _params.m;
    J(4, 4) = (Fy1_dv * cos(u.delta()) + Fy2_dv) / _params.m;
    J(4, 5) = -x.u() + (Fy1_dw * cos(u.delta()) + Fy2_dw) / _params.m;

    // Row 5:
    J(5, 3) = (_params.l_f * Fy1_du * cos(u.delta()) - _params.l_r * Fy2_du) / _params.I_z;
    J(5, 4) = (_params.l_f * Fy1_dv * cos(u.delta()) - _params.l_r * Fy2_dv) / _params.I_z;
    J(5, 5) = (_params.l_f * Fy1_dw * cos(u.delta()) - _params.l_r * Fy2_dw) / _params.I_z;

    return I + u.dt() * J;
}

Z DBMMK1::Model::predict_measure(const X &state)
{
    Z prediction;
    StateAcessor x(state);

    /* p_x   */ prediction(0) = x.p_x();
    /* p_y   */ prediction(1) = x.p_y();
    /* phi   */ prediction(2) = x.phi();
    /* omega */ prediction(3) = x.omega();

    return prediction;
}

EMatrix<double, MEASURE_DIM, STATE_DIM> DBMMK1::Model::get_measure_jacobian(const X &state)
{
    (void)state; // Supress the foking compilation warning like OK CMAKE YOURE SO INTELLIGENT stfu you silly ass bitch

    // The jacobian is literally trivial to calculate from the function avobe lmao :3
    EMatrix<double, MEASURE_DIM, STATE_DIM> J = EMatrix<double, MEASURE_DIM, STATE_DIM>::Zero();

    J(0, 0) = 1; // ∂(p_x) / ∂(p_x)
    J(1, 1) = 1; // ∂(p_y) / ∂(p_y)
    J(2, 2) = 1; // ∂(phi) / ∂(phi)
    J(3, 5) = 1; // ∂(omega) / ∂(omega)

    return J;
}
