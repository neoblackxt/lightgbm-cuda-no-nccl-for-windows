/*!
 * Minimal single-header stub of NVIDIA's <nccl.h> for LightGBM.
 *
 * Why this exists: NCCL has no official Windows distribution, but since the
 * multi-GPU CUDA feature (#6138) LightGBM's CMake unconditionally requires
 * the NCCL library even for single-GPU builds. Every NCCL call site in
 * LightGBM is only reached when `num_gpu > 1` (see Boosting::CreateBoosting
 * in src/boosting/boosting.cpp), and single-GPU training keeps
 * `nccl_communicator_ == nullptr` and skips all NCCL branches at runtime.
 *
 * This stub therefore only needs to satisfy the compiler and the linker.
 * The companion nccl_stub.cpp aborts loudly if any NCCL entry point is ever
 * called, so an accidental multi-GPU configuration fails fast instead of
 * silently producing wrong results.
 *
 * The ncclUniqueId layout keeps NCCL_UNIQUE_ID_BYTES == 128, matching the
 * real NCCL header, because LightGBM scatters it via Network::Allgather
 * using sizeof(ncclUniqueId).
 */
#ifndef NCCL_STUB_H_
#define NCCL_STUB_H_

#include <cuda_runtime.h>
#include <stddef.h>
#include <stdint.h>

#define NCCL_MAJOR(x) ((x) / 1000)
#define NCCL_MINOR(x) (((x) / 100) % 10)
#define NCCL_PATCH(x) ((x) % 100)
#define NCCL_VERSION_CODE 2700
#define NCCL_UNIQUE_ID_BYTES 128

#define NCCL_UNIQUE_ID_BYTES 128

typedef struct { char internal[NCCL_UNIQUE_ID_BYTES]; } ncclUniqueId;

typedef struct ncclComm* ncclComm_t;

typedef enum {
  ncclSuccess = 0,
  ncclUnhandledCudaError = 1,
  ncclSystemError = 2,
  ncclInternalError = 3,
  ncclInvalidArgument = 4,
  ncclInvalidUsage = 5,
  ncclRemoteError = 6,
  ncclInProgress = 7,
  ncclNumResults = 8
} ncclResult_t;

typedef enum {
  ncclInt8 = 0,
  ncclChar = 0,
  ncclUint8 = 1,
  ncclInt32 = 2,
  ncclInt = 2,
  ncclUint32 = 3,
  ncclInt64 = 4,
  ncclUint64 = 5,
  ncclFloat16 = 6,
  ncclHalf = 6,
  ncclFloat32 = 7,
  ncclFloat = 7,
  ncclFloat64 = 8,
  ncclDouble = 8,
  ncclBfloat16 = 9,
  ncclNumTypes = 10
} ncclDataType_t;

typedef enum {
  ncclSum = 0,
  ncclProd = 1,
  ncclMax = 2,
  ncclMin = 3,
  ncclAvg = 4,
  ncclNumOps = 5
} ncclRedOp_t;

/* group collective scheduling */
ncclResult_t ncclGroupStart(void);
ncclResult_t ncclGroupEnd(void);
ncclResult_t ncclGroupAbort(void);

/* communicator bootstrap */
ncclResult_t ncclGetUniqueId(ncclUniqueId* uniqueId);
ncclResult_t ncclCommInitRank(ncclComm_t* comm, int nranks, ncclUniqueId commId, int rank);
ncclResult_t ncclCommInitAll(ncclComm_t* comms, int ndev, const int* devlist);
ncclResult_t ncclCommDestroy(ncclComm_t comm);
ncclResult_t ncclCommAbort(ncclComm_t comm);
ncclResult_t ncclCommCount(const ncclComm_t comm, int* count);
ncclResult_t ncclCommCuDevice(const ncclComm_t comm, int* device);
ncclResult_t ncclCommUserRank(const ncclComm_t comm, int* rank);

/* collectives used by LightGBM */
ncclResult_t ncclAllReduce(const void* sendbuff, void* recvbuff, size_t count,
                           ncclDataType_t datatype, ncclRedOp_t op, ncclComm_t comm,
                           cudaStream_t stream);
ncclResult_t ncclBroadcast(const void* sendbuff, void* recvbuff, size_t count,
                           ncclDataType_t datatype, int root, ncclComm_t comm,
                           cudaStream_t stream);
ncclResult_t ncclSend(const void* sendbuff, size_t count, ncclDataType_t datatype,
                      int peer, ncclComm_t comm, cudaStream_t stream);
ncclResult_t ncclRecv(void* recvbuff, size_t count, ncclDataType_t datatype,
                      int peer, ncclComm_t comm, cudaStream_t stream);

/* diagnostics */
const char* ncclGetErrorString(ncclResult_t result);
const char* ncclGetVersion(void);

#endif  // NCCL_STUB_H_
