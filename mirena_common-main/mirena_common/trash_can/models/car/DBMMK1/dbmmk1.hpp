#ifndef dbmmk1_hpp
#define dbmmk1_hpp

#include "common_lib/ekf/extended_kalman_filter.hpp"

namespace DBMMK1 
{

    static constexpr int STATE_DIM = 6;
    static constexpr int CONTROL_DIM = 3;
    static constexpr int MEASURE_DIM = 4;

    typedef EMatrix<double, STATE_DIM, 1> X;
    typedef EMatrix<double, CONTROL_DIM, 1> U;
    typedef EMatrix<double, MEASURE_DIM, 1> Z;

    // Zero-Cost Wrapper for Vehicle State to be able to access elements by name
    class StateAcessor {
    public:
        // Constructor to initialize with a const reference to the state vector
        StateAcessor(const X& state)
            : _state(state) {}
    
        // Const getters for each element in the state vector
        const double& p_x() const { return _state(0); } // Global X position
        const double& p_y() const { return _state(1); } // Global Y position
        const double& phi() const { return _state(2); } // Heading angle (yaw)
        const double& u() const { return _state(3); } // Longitudinal velocity (in body frame)
        const double& v() const { return _state(4); } // Lateral velocity (in body frame)
        const double& omega() const { return _state(5); } // Yaw rate (angular velocity)
    
    private:
        const X& _state;
    };
    
    // Wrapper for Control Inputs
    class ControlAcessor {
    public:
        // Constructor to initialize with a const reference to the control input vector
        ControlAcessor(const U& control)
            : _control(control) {}

        // Const getters for each element in the control input vector
        const double& a() const { return _control(0); } // Longitudinal acceleration
        const double& delta() const { return _control(1); } // Steering angle (front wheels)
        const double& dt() const { return _control(2); } // Time step
    
    private:
        const U& _control;
    };
    
    struct Parameters{
        double l_f; /* Distance from centrer of mass to front axis */
        double k_f; /* Front axis equivalent sideslip stiffness */
        double l_r; /* Distance from centrer of mass to rear axis */
        double k_r; /* Real axis equivalent sideslip stiffness */
        double I_z; /* Yaw inertial of vehicle body */
        double m; /* Mass of the vehicle */
    };
    
    // Implementation of the Dynamic Bicycle Model Mark 1 using Forward Euler
    // Ready to be used in the EKM as both the measure and prediction model
    class Model: public mirena::StatePredictor<STATE_DIM, CONTROL_DIM>, public mirena::MeasurePredictor<STATE_DIM, MEASURE_DIM>
    {
    public:
    const Parameters _params;

    /////////////////////////////////////////////////////////////////////////////////
    // CONSTRUCTORS & FACTORIES
    /////////////////////////////////////////////////////////////////////////////////
 
    Model() = delete; /* The model must be parametrized */
    Model(Parameters params) : _params(params) {}

    // Build a Extended Kalman Filter using this model
    static mirena::ExtendedKalmanFilter<STATE_DIM, CONTROL_DIM, MEASURE_DIM> build_ekf(const Parameters &model_parameters);

    /////////////////////////////////////////////////////////////////////////////////
    // INTERFACES
    /////////////////////////////////////////////////////////////////////////////////
    
    // Predict next state based on current state + control input. Corresponds the "f" in theory
    X predict_state(
        const X &previous_state,
        const U &control) override;

    // Get jacobian evaluated at the given state assuming the control vector is constant. Corresponds to "F" in theory
    EMatrix<double, STATE_DIM, STATE_DIM> get_state_jacobian(
        const X &state,
        const U &control) override;

    // Predict next measure based on current state. Corresponds to "h" in theory
    EMatrix<double, MEASURE_DIM, 1> predict_measure(
        const X &state) override;
    
    // Get jacobian evaluated at the given state. Corresponds to "H" in theory
    EMatrix<double, MEASURE_DIM, STATE_DIM> get_measure_jacobian(
        const X &state) override;

    /////////////////////////////////////////////////////////////////////////////////
    // UTILS 
    /////////////////////////////////////////////////////////////////////////////////

    };

} // namespace mirena

#endif