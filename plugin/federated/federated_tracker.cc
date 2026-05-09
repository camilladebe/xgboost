/**
 * Copyright 2022-2024, XGBoost contributors
 */
#include "federated_tracker.h"

#include <grpcpp/security/server_credentials.h>  // for InsecureServerCredentials, ...
#include <grpcpp/server_builder.h>               // for ServerBuilder

#include <cstdint>    // for int32_t
#include <exception>  // for exception
#include <future>     // for future, async
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>     // for numeric_limits
#include <string>     // for string
#include <vector>

#include "../../src/common/io.h"          // for ReadAll
#include "../../src/common/json_utils.h"  // for RequiredArg

namespace xgboost::collective {
namespace federated {
grpc::Status FederatedService::Allgather(grpc::ServerContext*, AllgatherRequest const* request,
                                         AllgatherReply* reply) {
  handler_.Allgather(request->send_buffer().data(), request->send_buffer().size(),
                     reply->mutable_receive_buffer(), request->sequence_number(), request->rank());
  return grpc::Status::OK;
}

grpc::Status FederatedService::AllgatherV(grpc::ServerContext*, AllgatherVRequest const* request,
                                          AllgatherVReply* reply) {
  handler_.AllgatherV(request->send_buffer().data(), request->send_buffer().size(),
                      reply->mutable_receive_buffer(), request->sequence_number(), request->rank());
  return grpc::Status::OK;
}

grpc::Status FederatedService::Allreduce(grpc::ServerContext*, AllreduceRequest const* request,
                                         AllreduceReply* reply) {
  double client_time_max = 0.0;
  double server_aggregation_time = 0.0;
  handler_.Allreduce(request->send_buffer().data(), request->send_buffer().size(),
                     reply->mutable_receive_buffer(), request->sequence_number(), request->rank(),
                     static_cast<xgboost::ArrayInterfaceHandler::Type>(request->data_type()),
                     static_cast<xgboost::collective::Op>(request->reduce_operation()),
                     request->client_total_time_s(), &client_time_max, &server_aggregation_time);
  reply->set_server_aggregation_time_s(server_aggregation_time);
  reply->set_client_time_max_s(client_time_max);
  return grpc::Status::OK;
}

grpc::Status FederatedService::ReportTiming(grpc::ServerContext*, ReportTimingRequest const* request,
                                            ReportTimingReply*) {
  if (tracker_ != nullptr) {
    std::vector<TimingRecord> rows;
    rows.reserve(static_cast<std::size_t>(request->rows_size()));
    for (auto const& row : request->rows()) {
      rows.push_back(TimingRecord{row.iteration(), row.tree_id(), row.tree_node_id(),
                                  row.rank(), row.compute_time_s(), row.server_time_s(),
                                  row.communication_time_s()});
    }
    tracker_->AppendTimingRows(rows);
  }
  return grpc::Status::OK;
}

grpc::Status FederatedService::Broadcast(grpc::ServerContext*, BroadcastRequest const* request,
                                         BroadcastReply* reply) {
  handler_.Broadcast(request->send_buffer().data(), request->send_buffer().size(),
                     reply->mutable_receive_buffer(), request->sequence_number(), request->rank(),
                     request->root());
  return grpc::Status::OK;
}
}  // namespace federated

FederatedTracker::FederatedTracker(Json const& config) : Tracker{config} {
  auto is_secure = RequiredArg<Boolean const>(config, "federated_secure", __func__);
  if (is_secure) {
    StringView msg{"Empty certificate path."};
    server_key_path_ = RequiredArg<String const>(config, "server_key_path", __func__);
    CHECK(!server_key_path_.empty()) << msg;
    server_cert_file_ = RequiredArg<String const>(config, "server_cert_path", __func__);
    CHECK(!server_cert_file_.empty()) << msg;
    client_cert_file_ = RequiredArg<String const>(config, "client_cert_path", __func__);
    CHECK(!client_cert_file_.empty()) << msg;
  }

  bool const default_timing_enabled = false;
  std::string const default_timing_path;
  timing_enabled_ =
      OptionalArg<Boolean, bool>(config, "federated_timing.enabled", default_timing_enabled);
  timing_path_ =
      OptionalArg<String, std::string>(config, "federated_timing.path", default_timing_path);
}

std::future<Result> FederatedTracker::Run() {
  return std::async(std::launch::async, [this]() {
    std::string const server_address = "0.0.0.0:" + std::to_string(this->port_);
    xgboost::collective::federated::FederatedService service{
        static_cast<std::int32_t>(this->n_workers_), this};
    grpc::ServerBuilder builder;

    if (timing_enabled_ && !timing_path_.empty()) {
      std::lock_guard<std::mutex> lk(timing_mutex_);
      timing_stream_.open(timing_path_, std::ios::out | std::ios::app);
      CHECK(timing_stream_.is_open()) << "Failed to open timing CSV: " << timing_path_;
      if (std::filesystem::exists(timing_path_) && std::filesystem::file_size(timing_path_) > 0) {
        timing_header_written_ = true;
      }
      if (!timing_header_written_) {
        timing_stream_ << "iteration,tree_id,tree_node_id,rank,compute_time_s,server_time_s,communication_time_s\n";
        timing_stream_.flush();
        timing_header_written_ = true;
      }
    }

    if (this->server_cert_file_.empty()) {
      builder.SetMaxReceiveMessageSize(std::numeric_limits<std::int32_t>::max());
      if (this->port_ == 0) {
        builder.AddListeningPort(server_address, grpc::InsecureServerCredentials(), &port_);
      } else {
        builder.AddListeningPort(server_address, grpc::InsecureServerCredentials());
      }
      builder.RegisterService(&service);
      LOG(CONSOLE) << "Insecure federated server listening on " << server_address << ", world size "
                   << this->n_workers_;
    } else {
      auto options = grpc::SslServerCredentialsOptions(
          GRPC_SSL_REQUEST_AND_REQUIRE_CLIENT_CERTIFICATE_AND_VERIFY);
      options.pem_root_certs = xgboost::common::ReadAll(client_cert_file_);
      auto key = grpc::SslServerCredentialsOptions::PemKeyCertPair();
      key.private_key = xgboost::common::ReadAll(server_key_path_);
      key.cert_chain = xgboost::common::ReadAll(server_cert_file_);
      options.pem_key_cert_pairs.push_back(key);
      builder.SetMaxReceiveMessageSize(std::numeric_limits<std::int32_t>::max());
      if (this->port_ == 0) {
        builder.AddListeningPort(server_address, grpc::SslServerCredentials(options), &port_);
      } else {
        builder.AddListeningPort(server_address, grpc::SslServerCredentials(options));
      }
      builder.RegisterService(&service);
      LOG(CONSOLE) << "Federated server listening on " << server_address << ", world size "
                   << n_workers_;
    }

    try {
      server_ = builder.BuildAndStart();
      ready_ = true;
      server_->Wait();
    } catch (std::exception const& e) {
      return collective::Fail(std::string{e.what()});
    }

    ready_ = false;
    return collective::Success();
  });
}

FederatedTracker::~FederatedTracker() = default;

Result FederatedTracker::Shutdown() {
  auto rc = this->WaitUntilReady();
  SafeColl(rc);

  try {
    server_->Shutdown();
  } catch (std::exception const& e) {
    return Fail("Failed to shutdown:" + std::string{e.what()});
  }

  return Success();
}

void FederatedTracker::AppendTimingRows(std::vector<TimingRecord> const& rows) const {
  if (!timing_enabled_ || timing_path_.empty() || rows.empty()) {
    return;
  }

  std::lock_guard<std::mutex> lk(timing_mutex_);
  if (!timing_stream_.is_open()) {
    timing_stream_.open(timing_path_, std::ios::out | std::ios::app);
    CHECK(timing_stream_.is_open()) << "Failed to open timing CSV: " << timing_path_;
  }

  timing_stream_ << std::setprecision(17);
  for (auto const& row : rows) {
    timing_stream_ << row.iteration << ',' << row.tree_id << ',' << row.tree_node_id << ','
                   << row.rank << ',' << row.compute_time_s << ',' << row.server_time_s << ','
                   << row.communication_time_s << '\n';
  }
  timing_stream_.flush();
}

[[nodiscard]] Json FederatedTracker::WorkerArgs() const {
  auto rc = this->WaitUntilReady();
  SafeColl(rc);

  std::string host;
  rc = GetHostAddress(&host);
  SafeColl(rc);
  Json args{Object{}};
  args["dmlc_tracker_uri"] = String{host};
  args["dmlc_tracker_port"] = this->Port();
  return args;
}
}  // namespace xgboost::collective
