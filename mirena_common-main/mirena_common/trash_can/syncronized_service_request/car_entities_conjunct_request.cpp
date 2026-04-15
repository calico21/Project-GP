#include "car_entities_conjunct_request.hpp"

using namespace mirena;
using namespace std::chrono_literals;

CarEntitiesConjunctRequest::CarEntitiesConjunctRequest(
    GetCarClient::SharedPtr car_client,
    GetCarSrv::Request::SharedPtr car_request,
    GetEntitiesClient::SharedPtr entities_client,
    GetEntitiesSrv::Request::SharedPtr entities_request,
    std::chrono::milliseconds timeout,
    rclcpp::Node &requester,
    std::function<void(CarEntitiesConjunctRequest &)> on_complete) : _is_alive(true),
                                                                     _timeout_timer(requester.create_wall_timer(timeout, [this]
                                                                                                                { this->cancel_request(); })),
                                                                     _on_complete(on_complete),
                                                                     _car_client(car_client),
                                                                     _car_future_id(car_client->async_send_request(car_request, [this](GetCarClient::SharedFuture future)
                                                                                                                   { this->_on_car_future_resolved(future); })),
                                                                     _entities_client(entities_client),
                                                                     _entities_future_id(entities_client->async_send_request(entities_request, [this](GetEntitiesClient::SharedFuture future)
                                                                                                                             { this->_on_entities_future_resolved(future); }))

{
}

void CarEntitiesConjunctRequest::cancel_request()
{
    _processing_lock.lock();

    if (!_is_alive)
    {
        return;
    }
    _timeout_timer->cancel();
    if (!_car_ready)
    {
        _car_client->remove_pending_request(_car_future_id);
    }
    if (!_entities_ready)
    {
        _entities_client->remove_pending_request(_entities_future_id);
    }
    _is_alive = false;

    _processing_lock.unlock();
}

std::shared_ptr<CarEntitiesConjunctRequest> mirena::CarEntitiesConjunctRequest::make_request(GetCarClient::SharedPtr car_client, GetCarSrv::Request::SharedPtr car_request, GetEntitiesClient::SharedPtr entities_client, GetEntitiesSrv::Request::SharedPtr entities_request, std::chrono::milliseconds timeout, rclcpp::Node &requester, std::function<void(CarEntitiesConjunctRequest &)> on_complete)
{
    if (!car_client->wait_for_service(100ms) | !entities_client->wait_for_service(100ms))
    {
        RCLCPP_WARN(requester.get_logger(), "Services are not ready - cannot request car and entities");
        return 0;
    }
    return std::make_shared<CarEntitiesConjunctRequest>(car_client, car_request, entities_client, entities_request, timeout, requester, on_complete);
}

bool CarEntitiesConjunctRequest::is_alive()
{
    return _is_alive;
}

void mirena::CarEntitiesConjunctRequest::_on_timeout()
{
    cancel_request();
}

void mirena::CarEntitiesConjunctRequest::_on_car_future_resolved(GetCarClient::SharedFuture future)
{
    _processing_lock.lock();

    _car_ready = true;
    _on_either_future_resolved();

    _processing_lock.unlock();
}

void mirena::CarEntitiesConjunctRequest::_on_entities_future_resolved(GetEntitiesClient::SharedFuture future)
{
    _processing_lock.lock();

    _entities_ready = true;
    _on_either_future_resolved();

    _processing_lock.unlock();
}

void mirena::CarEntitiesConjunctRequest::_on_either_future_resolved()
{
    if (!(_car_ready & _entities_ready))
    {
        return;
    }
    _on_complete(*this);
    _is_alive = false;
};