/**
 * Copyright 2023, XGBoost contributors
 */
#include "federated_coll.h"

#include <federated.grpc.pb.h>
#include <federated.pb.h>
#include <chrono>

#include <algorithm>  // for copy_n

#include "../../src/collective/allgather.h"
#include "../../src/common/common.h"    // for AssertGPUSupport
#include "federated_comm.h"             // for FederatedComm
#include "xgboost/collective/result.h"  // for Result

namespace xgboost::collective {
namespace {
[[nodiscard]] Result GetGRPCResult(std::string const &name, grpc::Status const &status) {
  return Fail(name + " RPC failed. " + std::to_string(status.error_code()) + ": " +
              status.error_message());
}

[[nodiscard]] Result BroadcastImpl(Comm const &comm, std::uint64_t *sequence_number,
                                   common::Span<std::int8_t> data, std::int32_t root) {
  using namespace federated;  // NOLINT

  auto fed = dynamic_cast<FederatedComm const *>(&comm);
  CHECK(fed);
  auto stub = fed->Handle();

  BroadcastRequest request;
  request.set_sequence_number((*sequence_number)++);
  request.set_rank(comm.Rank());
  if (comm.Rank() != root) {
    request.set_send_buffer(nullptr, 0);
  } else {
    request.set_send_buffer(data.data(), data.size());
  }
  request.set_root(root);

  BroadcastReply reply;
  grpc::ClientContext context;
  context.set_wait_for_ready(true);
  grpc::Status status = stub->Broadcast(&context, request, &reply);
  if (!status.ok()) {
    return GetGRPCResult("Broadcast", status);
  }
  if (comm.Rank() != root) {
    auto const &r = reply.receive_buffer();
    std::copy_n(r.cbegin(), r.size(), data.data());
  }

  return Success();
}

[[nodiscard]] Result ReportTimingImpl(Comm const &comm, std::vector<TimingRecord> const &rows) {
  using namespace federated;  // NOLINT

  auto fed = dynamic_cast<FederatedComm const *>(&comm);
  CHECK(fed);
  auto stub = fed->Handle();

  ReportTimingRequest request;
  for (auto const &row : rows) {
    auto *timing_row = request.add_rows();
    timing_row->set_iteration(row.iteration);
    timing_row->set_tree_id(row.tree_id);
    timing_row->set_tree_node_id(row.tree_node_id);
    timing_row->set_rank(row.rank);
    timing_row->set_compute_time_s(row.compute_time_s);
    timing_row->set_client_time_s(row.client_time_s);
    timing_row->set_server_time_s(row.server_time_s);
    timing_row->set_communication_time_s(row.communication_time_s);
  }

  ReportTimingReply reply;
  grpc::ClientContext context;
  context.set_wait_for_ready(true);
  grpc::Status status = stub->ReportTiming(&context, request, &reply);
  if (!status.ok()) {
    return GetGRPCResult("ReportTiming", status);
  }
  return Success();
}
}  // namespace

#if !defined(XGBOOST_USE_CUDA)
Coll *FederatedColl::MakeCUDAVar() {
  common::AssertGPUSupport();
  return nullptr;
}
#endif

[[nodiscard]] Result FederatedColl::Allreduce(Comm const &comm, common::Span<std::int8_t> data,
                                              ArrayInterfaceHandler::Type type, Op op) {
  using namespace federated;  // NOLINT
  auto fed = dynamic_cast<FederatedComm const *>(&comm);
  CHECK(fed);
  auto stub = fed->Handle();

  AllreduceRequest request;
  request.set_sequence_number(sequence_number_++);
  request.set_rank(comm.Rank());
  request.set_send_buffer(data.data(), data.size());
  request.set_data_type(static_cast<::xgboost::collective::federated::DataType>(type));
  request.set_reduce_operation(static_cast<::xgboost::collective::federated::ReduceOperation>(op));

  AllreduceReply reply;
  grpc::ClientContext context;
  context.set_wait_for_ready(true);
  auto send_start = std::chrono::steady_clock::now();
  double client_time_s = 0.0;
  if (has_last_allreduce_receive_time_) {
    std::chrono::duration<double> client_duration =
        send_start - last_allreduce_receive_time_;
    client_time_s = client_duration.count();
  }
  request.set_client_total_time_s(client_time_s);
  grpc::Status status = stub->Allreduce(&context, request, &reply);
  auto rpc_end = std::chrono::steady_clock::now();
  if (!status.ok()) {
    return GetGRPCResult("Allreduce", status);
  }
  std::chrono::duration<double> rpc_duration = rpc_end - send_start;
  double client_allreduce_time_s = rpc_duration.count();
  auto const &r = reply.receive_buffer();
  std::copy_n(r.cbegin(), r.size(), data.data());
  last_allreduce_receive_time_ = std::chrono::steady_clock::now();
  has_last_allreduce_receive_time_ = true;
  double server_agg_s = reply.server_aggregation_time_s();
  double server_handler_s = reply.server_handler_time_s();
  double client_max_s = reply.client_time_max_s();
  double communication_s = std::max(0.0, client_allreduce_time_s - server_handler_s);
  last_client_time_s_ = client_time_s;
  last_server_aggregation_time_s_ = server_agg_s;
  last_server_handler_time_s_ = server_handler_s;
  last_communication_time_s_ = communication_s;
  LOG(INFO) << "[AllReduce rank=" << comm.Rank() << "] client_time_s=" << client_time_s
            << " client_allreduce_s=" << client_allreduce_time_s
            << " client_max_s=" << client_max_s << " server_agg_s=" << server_agg_s
            << " server_handler_s=" << server_handler_s << " communication_s=" << communication_s;
  return Success();
}

[[nodiscard]] Result FederatedColl::Broadcast(Comm const &comm, common::Span<std::int8_t> data,
                                              std::int32_t root) {
  return BroadcastImpl(comm, &this->sequence_number_, data, root);
}

[[nodiscard]] Result FederatedColl::Allgather(Comm const &comm, common::Span<std::int8_t> data) {
  using namespace federated;  // NOLINT
  auto fed = dynamic_cast<FederatedComm const *>(&comm);
  CHECK(fed);
  auto stub = fed->Handle();
  auto size = data.size_bytes() / comm.World();

  auto offset = comm.Rank() * size;
  auto segment = data.subspan(offset, size);

  AllgatherRequest request;
  request.set_sequence_number(sequence_number_++);
  request.set_rank(comm.Rank());
  request.set_send_buffer(segment.data(), segment.size());

  AllgatherReply reply;
  grpc::ClientContext context;
  context.set_wait_for_ready(true);
  grpc::Status status = stub->Allgather(&context, request, &reply);

  if (!status.ok()) {
    return GetGRPCResult("Allgather", status);
  }
  auto const &r = reply.receive_buffer();
  std::copy_n(r.cbegin(), r.size(), data.begin());
  return Success();
}

[[nodiscard]] Result FederatedColl::AllgatherV(Comm const &comm,
                                               common::Span<std::int8_t const> data,
                                               common::Span<std::int64_t const>,
                                               common::Span<std::int64_t>,
                                               common::Span<std::int8_t> recv, AllgatherVAlgo) {
  using namespace federated;  // NOLINT

  auto fed = dynamic_cast<FederatedComm const *>(&comm);
  CHECK(fed);
  auto stub = fed->Handle();

  AllgatherVRequest request;
  request.set_sequence_number(sequence_number_++);
  request.set_rank(comm.Rank());
  request.set_send_buffer(data.data(), data.size());

  AllgatherVReply reply;
  grpc::ClientContext context;
  context.set_wait_for_ready(true);
  grpc::Status status = stub->AllgatherV(&context, request, &reply);
  if (!status.ok()) {
    return GetGRPCResult("AllgatherV", status);
  }
  std::string const &r = reply.receive_buffer();
  CHECK_EQ(r.size(), recv.size());
  std::copy_n(r.cbegin(), r.size(), recv.begin());
  return Success();
}

[[nodiscard]] Result FederatedColl::ReportTiming(Comm const &comm,
                                                 std::vector<TimingRecord> const &rows) {
  return ReportTimingImpl(comm, rows);
}
}  // namespace xgboost::collective
