/**
 * Copyright 2022-2023, XGBoost contributors
 */
#pragma once
#include <federated.grpc.pb.h>  // for Server

#include <future>  // for future
#include <fstream>
#include <memory>  // for unique_ptr
#include <mutex>
#include <string>  // for string
#include <vector>

#include "../../src/collective/coll.h"
#include "../../src/collective/in_memory_handler.h"
#include "../../src/collective/tracker.h"  // for Tracker
#include "xgboost/collective/result.h"     // for Result
#include "xgboost/json.h"                  // for Json

namespace xgboost::collective {
class FederatedTracker;

namespace federated {

class FederatedService final : public Federated::Service {
 public:
  explicit FederatedService(std::int32_t world_size, xgboost::collective::FederatedTracker* tracker)
      : handler_{world_size}, tracker_{tracker} {}

  grpc::Status Allgather(grpc::ServerContext* context, AllgatherRequest const* request,
                         AllgatherReply* reply) override;

  grpc::Status AllgatherV(grpc::ServerContext* context, AllgatherVRequest const* request,
                          AllgatherVReply* reply) override;

  grpc::Status Allreduce(grpc::ServerContext* context, AllreduceRequest const* request,
                         AllreduceReply* reply) override;

  grpc::Status ReportTiming(grpc::ServerContext* context, ReportTimingRequest const* request,
                            ReportTimingReply* reply) override;

  grpc::Status Broadcast(grpc::ServerContext* context, BroadcastRequest const* request,
                         BroadcastReply* reply) override;

 private:
  xgboost::collective::InMemoryHandler handler_;
  xgboost::collective::FederatedTracker* tracker_{nullptr};
};
};  // namespace federated

class FederatedTracker : public collective::Tracker {
  std::unique_ptr<grpc::Server> server_;
  std::string server_key_path_;
  std::string server_cert_file_;
  std::string client_cert_file_;
  bool timing_enabled_{false};
  std::string timing_path_;
  mutable std::mutex timing_mutex_;
  mutable std::ofstream timing_stream_;
  mutable bool timing_header_written_{false};

 public:
  /**
   * @brief CTOR
   *
   * @param config Configuration, other than the base configuration from Tracker, we have:
   *
   * - federated_secure: bool whether this is a secure server.
   * - server_key_path: path to the key.
   * - server_cert_path: certificate path.
   * - client_cert_path: certificate path for client.
   */
  explicit FederatedTracker(Json const& config);
  ~FederatedTracker() override;
  std::future<Result> Run() override;

  [[nodiscard]] Json WorkerArgs() const override;
  [[nodiscard]] Result Shutdown();

  void AppendTimingRows(std::vector<TimingRecord> const& rows) const;
};
}  // namespace xgboost::collective
