#pragma once
#include "mirena_planning.hpp"

#include <mutex>

// <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<   WARNING  <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
// This is both untestes and unused
// Originally it was used to syncronize the request of 2 services which is actually obscenely hard for what it is.
// We decided to transition to a pub-sub based approach almost immediately after having to do this


// GOD I LOVE ASYNC OMG GOOOOOOD HOW MUCH I LOVE ASYNC SHIT. IT IS NOT MISERABLE AND OVERLY COMPLICATED GOOOD I LOVE ASYNC AND ALL THE BOILERPLATE IT ENTAILS 
namespace mirena {
    class CarEntitiesConjunctRequest{
        using GetCarSrv=mirena_common::srv::GetCar;
        using GetEntitiesSrv=mirena_common::srv::GetEntities;
        using GetCarClient=rclcpp::Client<GetCarSrv>;
        using GetEntitiesClient=rclcpp::Client<GetEntitiesSrv>;
        
        bool _is_alive; 
        rclcpp::TimerBase::SharedPtr _timeout_timer;
        std::function<void(CarEntitiesConjunctRequest &)> _on_complete;
        std::mutex _processing_lock;

        bool _car_ready = false;
        bool _entities_ready = false;

        GetCarClient::SharedPtr _car_client;
        GetCarClient::SharedFutureAndRequestId _car_future_id;
        GetCarSrv::Response::SharedPtr get_car_response();
        
        GetEntitiesClient::SharedPtr _entities_client;
        GetEntitiesClient::SharedFutureAndRequestId _entities_future_id;
        GetCarSrv::Response::SharedPtr entities_response;

        // Retruns nullptr if couldnt make a request 
        static std::shared_ptr<CarEntitiesConjunctRequest> make_request(
            GetCarClient::SharedPtr car_client,
            GetCarSrv::Request::SharedPtr car_request, 
            GetEntitiesClient::SharedPtr entities_client,
            GetEntitiesSrv::Request::SharedPtr entities_request,
            std::chrono::milliseconds timeout,
            rclcpp::Node& requester,
            std::function<void(CarEntitiesConjunctRequest&)> on_complete
        );

        void cancel_request();
        bool is_alive();

        private:
        CarEntitiesConjunctRequest() = delete;
        CarEntitiesConjunctRequest(
            GetCarClient::SharedPtr car_client,
            GetCarSrv::Request::SharedPtr car_request, 
            GetEntitiesClient::SharedPtr entities_client,
            GetEntitiesSrv::Request::SharedPtr entities_request,
            std::chrono::milliseconds timeout,
            rclcpp::Node& requester,
            std::function<void(CarEntitiesConjunctRequest&)> on_complete
        );
        ~CarEntitiesConjunctRequest(){
            cancel_request();
        };
        void _on_timeout();
        void _on_car_future_resolved(GetCarClient::SharedFuture future);
        void _on_entities_future_resolved(GetEntitiesClient::SharedFuture future);
        void _on_either_future_resolved();
    };
}