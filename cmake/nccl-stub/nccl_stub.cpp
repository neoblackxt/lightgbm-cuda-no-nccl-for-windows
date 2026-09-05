/*!
 * Stub implementation of the NCCL entry points LightGBM can reference.
 * See nccl.h for the rationale. Anything but ncclGetErrorString /
 * ncclGetVersion aborts: these functions must never run on a single-GPU
 * build, because LightGBM guards every NCCL branch with
 * `nccl_communicator != nullptr`, which only becomes non-null when
 * `num_gpu > 1`.
 */
#include <nccl.h>

#include <cstdio>
#include <cstdlib>

static void nccl_stub_fail(const char* fn) {
  std::fprintf(stderr,
               "[LightGBM] FATAL: %s was called, but this is a stub-NCCL build "
               "(single-GPU CUDA only). Multi-GPU training requires a real NCCL "
               "build on Linux.\n", fn);
  std::abort();
}

ncclResult_t ncclGroupStart(void) { nccl_stub_fail("ncclGroupStart"); return ncclSuccess; }
ncclResult_t ncclGroupEnd(void) { nccl_stub_fail("ncclGroupEnd"); return ncclSuccess; }
ncclResult_t ncclGroupAbort(void) { nccl_stub_fail("ncclGroupAbort"); return ncclSuccess; }

ncclResult_t ncclGetUniqueId(ncclUniqueId* uniqueId) {
  nccl_stub_fail("ncclGetUniqueId");
  if (uniqueId != nullptr) { uniqueId->internal[0] = '\0'; }
  return ncclSuccess;
}

ncclResult_t ncclCommInitRank(ncclComm_t* comm, int nranks, ncclUniqueId commId, int rank) {
  (void)commId;
  nccl_stub_fail("ncclCommInitRank");
  if (comm != nullptr) { *comm = nullptr; }
  (void)nranks; (void)rank;
  return ncclSuccess;
}

ncclResult_t ncclCommInitAll(ncclComm_t* comms, int ndev, const int* devlist) {
  nccl_stub_fail("ncclCommInitAll");
  if (comms != nullptr) { for (int i = 0; i < ndev; ++i) { comms[i] = nullptr; } }
  (void)devlist;
  return ncclSuccess;
}

ncclResult_t ncclCommDestroy(ncclComm_t comm) { nccl_stub_fail("ncclCommDestroy"); (void)comm; return ncclSuccess; }
ncclResult_t ncclCommAbort(ncclComm_t comm) { nccl_stub_fail("ncclCommAbort"); (void)comm; return ncclSuccess; }

ncclResult_t ncclCommCount(const ncclComm_t comm, int* count) {
  nccl_stub_fail("ncclCommCount"); (void)comm;
  if (count != nullptr) { *count = 0; }
  return ncclSuccess;
}

ncclResult_t ncclCommCuDevice(const ncclComm_t comm, int* device) {
  nccl_stub_fail("ncclCommCuDevice"); (void)comm;
  if (device != nullptr) { *device = -1; }
  return ncclSuccess;
}

ncclResult_t ncclCommUserRank(const ncclComm_t comm, int* rank) {
  nccl_stub_fail("ncclCommUserRank"); (void)comm;
  if (rank != nullptr) { *rank = -1; }
  return ncclSuccess;
}

ncclResult_t ncclAllReduce(const void* sendbuff, void* recvbuff, size_t count,
                           ncclDataType_t datatype, ncclRedOp_t op, ncclComm_t comm,
                           cudaStream_t stream) {
  nccl_stub_fail("ncclAllReduce");
  (void)sendbuff; (void)recvbuff; (void)count; (void)datatype; (void)op; (void)comm; (void)stream;
  return ncclSuccess;
}

ncclResult_t ncclBroadcast(const void* sendbuff, void* recvbuff, size_t count,
                           ncclDataType_t datatype, int root, ncclComm_t comm,
                           cudaStream_t stream) {
  nccl_stub_fail("ncclBroadcast");
  (void)sendbuff; (void)recvbuff; (void)count; (void)datatype; (void)root; (void)comm; (void)stream;
  return ncclSuccess;
}

ncclResult_t ncclSend(const void* sendbuff, size_t count, ncclDataType_t datatype,
                      int peer, ncclComm_t comm, cudaStream_t stream) {
  nccl_stub_fail("ncclSend");
  (void)sendbuff; (void)count; (void)datatype; (void)peer; (void)comm; (void)stream;
  return ncclSuccess;
}

ncclResult_t ncclRecv(void* recvbuff, size_t count, ncclDataType_t datatype,
                      int peer, ncclComm_t comm, cudaStream_t stream) {
  nccl_stub_fail("ncclRecv");
  (void)recvbuff; (void)count; (void)datatype; (void)peer; (void)comm; (void)stream;
  return ncclSuccess;
}

const char* ncclGetErrorString(ncclResult_t result) {
  (void)result;
  return "stub NCCL build (single-GPU only)";
}

const char* ncclGetVersion(void) { return "2.27.0-stub"; }
