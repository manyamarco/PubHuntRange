/*
 * This file is part of the VanitySearch distribution (https://github.com/JeanLucPons/VanitySearch).
 * Copyright (c) 2019 Jean Luc PONS.
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, version 3.
 *
 * This program is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
 * General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program. If not, see <http://www.gnu.org/licenses/>.
*/

#include "GPUEngine.h"
#include <ctime>
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <stdint.h>
#include "../Timer.h"

#include "GPUMath.h"
#include "GPUHash.h"
#include "GPUCompute.h"

// ---------------------------------------------------------------------------------------

static const char* __cudaRandGetErrorEnum(curandStatus_t error) {
	switch (error) {
	case CURAND_STATUS_SUCCESS:
		return "CURAND_STATUS_SUCCESS";

	case CURAND_STATUS_VERSION_MISMATCH:
		return "CURAND_STATUS_VERSION_MISMATCH";

	case CURAND_STATUS_NOT_INITIALIZED:
		return "CURAND_STATUS_NOT_INITIALIZED";

	case CURAND_STATUS_ALLOCATION_FAILED:
		return "CURAND_STATUS_ALLOCATION_FAILED";

	case CURAND_STATUS_TYPE_ERROR:
		return "CURAND_STATUS_TYPE_ERROR";

	case CURAND_STATUS_OUT_OF_RANGE:
		return "CURAND_STATUS_OUT_OF_RANGE";

	case CURAND_STATUS_LENGTH_NOT_MULTIPLE:
		return "CURAND_STATUS_LENGTH_NOT_MULTIPLE";

	case CURAND_STATUS_DOUBLE_PRECISION_REQUIRED:
		return "CURAND_STATUS_DOUBLE_PRECISION_REQUIRED";

	case CURAND_STATUS_LAUNCH_FAILURE:
		return "CURAND_STATUS_LAUNCH_FAILURE";

	case CURAND_STATUS_PREEXISTING_FAILURE:
		return "CURAND_STATUS_PREEXISTING_FAILURE";

	case CURAND_STATUS_INITIALIZATION_FAILED:
		return "CURAND_STATUS_INITIALIZATION_FAILED";

	case CURAND_STATUS_ARCH_MISMATCH:
		return "CURAND_STATUS_ARCH_MISMATCH";

	case CURAND_STATUS_INTERNAL_ERROR:
		return "CURAND_STATUS_INTERNAL_ERROR";
	}

	return "<unknown>";
}

inline void __cudaRandSafeCall(curandStatus_t err, const char* file, const int line)
{
	if (CURAND_STATUS_SUCCESS != err)
	{
		fprintf(stderr, "CudaRandSafeCall() failed at %s:%i : %s\n", file, line, __cudaRandGetErrorEnum(err));
		exit(-1);
	}
	return;
}

inline void __cudaSafeCall(cudaError err, const char* file, const int line)
{
	if (cudaSuccess != err)
	{
		fprintf(stderr, "cudaSafeCall() failed at %s:%i : %s\n", file, line, cudaGetErrorString(err));
		exit(-1);
	}
	return;
}

#define CudaRandSafeCall( err ) __cudaRandSafeCall( err, __FILE__, __LINE__ )
#define CudaSafeCall( err ) __cudaSafeCall( err, __FILE__, __LINE__ )

// ---------------------------------------------------------------------------------------

__global__ void compute_hash(uint64_t* keys, uint32_t* hash160, int numHash160, uint32_t maxFound, uint32_t* found, uint64_t startRange, uint64_t endRange)
{
    int id = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
    keys[id] = startRange + keys[id] % (endRange - startRange + 1);

    ComputeHash(keys + id, hash160, numHash160, maxFound, found);
}

// ---------------------------------------------------------------------------------------

using namespace std;

int _ConvertSMVer2Cores(int major, int minor)
{

	// Defines for GPU Architecture types (using the SM version to determine
	// the # of cores per SM
	typedef struct {
		int SM;  // 0xMm (hexidecimal notation), M = SM Major version,
		// and m = SM minor version
		int Cores;
	} sSMtoCores;

	sSMtoCores nGpuArchCoresPerSM[] = {
		{0x20, 32}, // Fermi Generation (SM 2.0) GF100 class
		{0x21, 48}, // Fermi Generation (SM 2.1) GF10x class
		{0x30, 192},
		{0x32, 192},
		{0x35, 192},
		{0x37, 192},
		{0x50, 128},
		{0x52, 128},
		{0x53, 128},
		{0x60,  64},
		{0x61, 128},
		{0x62, 128},
		{0x70,  64},
		{0x72,  64},
		{0x75,  64},
		{0x80,  64},
		{0x86, 128},
		{0x87, 128},
		{0x89, 128},
		{0x90, 128},
		{-1, -1}
	};

	int index = 0;

	while (nGpuArchCoresPerSM[index].SM != -1) {
		if (nGpuArchCoresPerSM[index].SM == ((major << 4) + minor)) {
			return nGpuArchCoresPerSM[index].Cores;
		}

		index++;
	}

	return 0;

}

// ----------------------------------------------------------------------------

GPUEngine::GPUEngine(int nbThreadGroup, int nbThreadPerGroup, int gpuId, uint32_t maxFound, const uint32_t* hash160, int numHash160, uint64_t startRange, uint64_t endRange, uint64_t rndOffset)
{

	// Initialise CUDA
	this->nbThreadPerGroup = nbThreadPerGroup;
	this->numHash160 = numHash160;
	this->startRange = startRange;
    this->endRange = endRange;

	initialised = false;

	int deviceCount = 0;
	CudaSafeCall(cudaGetDeviceCount(&deviceCount));

	// This function call returns 0 if there are no CUDA capable devices.
	if (deviceCount == 0) {
		printf("GPUEngine: There are no available device(s) that support CUDA\n");
		exit(-1);
	}

	CudaSafeCall(cudaSetDevice(gpuId));

	cudaDeviceProp deviceProp;
	CudaSafeCall(cudaGetDeviceProperties(&deviceProp, gpuId));

	if (nbThreadGroup == -1)
		nbThreadGroup = deviceProp.multiProcessorCount * 8;

	this->nbThread = nbThreadGroup * nbThreadPerGroup;
	this->maxFound = maxFound;
	this->outputSize = (maxFound * ITEM_SIZE_A + 4);

	char tmp[512];
	sprintf(tmp, "GPU #%d %s (%dx%d cores) Grid(%dx%d)",
		gpuId, deviceProp.name, deviceProp.multiProcessorCount,
		_ConvertSMVer2Cores(deviceProp.major, deviceProp.minor),
		nbThread / nbThreadPerGroup,
		nbThreadPerGroup);
	deviceName = std::string(tmp);

	// Prefer L1 (We do not use __shared__ at all)
	CudaSafeCall(cudaDeviceSetCacheConfig(cudaFuncCachePreferL1));

	// Note: the large per-thread stack limit used by VanitySearch is not
	// needed here. This kernel is straight-line hashing with no deep call
	// chain or recursion, so we leave the default (small) stack, which avoids
	// reserving gigabytes of local-memory backing store.

	// Allocate memory (two key buffers for RNG / compute overlap)
	CudaSafeCall(cudaMalloc((void**)&inputKey[0], nbThread * 4 * sizeof(uint64_t)));
	CudaSafeCall(cudaMalloc((void**)&inputKey[1], nbThread * 4 * sizeof(uint64_t)));
	keyBuf = 0;

	CudaSafeCall(cudaMalloc((void**)&outputBuffer, outputSize));
	CudaSafeCall(cudaHostAlloc(&outputBufferPinned, outputSize, cudaHostAllocWriteCombined | cudaHostAllocMapped));

	int K_SIZE = 5;

	CudaSafeCall(cudaMalloc((void**)&inputHash, numHash160 * K_SIZE * sizeof(uint32_t)));
	CudaSafeCall(cudaHostAlloc(&inputHashPinned, numHash160 * K_SIZE * sizeof(uint32_t), cudaHostAllocWriteCombined | cudaHostAllocMapped));

	memcpy(inputHashPinned, hash160, numHash160 * K_SIZE * sizeof(uint32_t));

	CudaSafeCall(cudaMemcpy(inputHash, inputHashPinned, numHash160 * K_SIZE * sizeof(uint32_t), cudaMemcpyHostToDevice));
	CudaSafeCall(cudaFreeHost(inputHashPinned));
	inputHashPinned = NULL;

	// cuda-rand. Two independent non-blocking streams let the RNG stream and
	// the compute stream run concurrently.
	CudaSafeCall(cudaStreamCreateWithFlags(&rngStream, cudaStreamNonBlocking));
	CudaSafeCall(cudaStreamCreateWithFlags(&computeStream, cudaStreamNonBlocking));
	CudaRandSafeCall(curandCreateGenerator(&prngGPU, CURAND_RNG_QUASI_SCRAMBLED_SOBOL64));
	// Advance the quasi-random sequence past the keys already scanned in a
	// previous run (rndOffset) so a resumed search keeps exploring fresh
	// keys instead of repeating the same Sobol region.
	CudaRandSafeCall(curandSetGeneratorOffset(prngGPU, rndOffset + (uint64_t)std::time(0)));
	CudaRandSafeCall(curandSetStream(prngGPU, rngStream));

	// Prime the first key buffer so the very first Step has data ready.
	Randomize(keyBuf);
	CudaSafeCall(cudaStreamSynchronize(rngStream));

	CudaSafeCall(cudaGetLastError());

	initialised = true;

}

// ----------------------------------------------------------------------------

int GPUEngine::GetGroupSize()
{
	return GRP_SIZE;
}

// ----------------------------------------------------------------------------

void GPUEngine::PrintCudaInfo()
{
	const char* sComputeMode[] = {
		"Multiple host threads",
		"Only one host thread",
		"No host thread",
		"Multiple process threads",
		"Unknown",
		NULL
	};

	int deviceCount = 0;
	CudaSafeCall(cudaGetDeviceCount(&deviceCount));

	// This function call returns 0 if there are no CUDA capable devices.
	if (deviceCount == 0) {
		printf("GPUEngine: There are no available device(s) that support CUDA\n");
		return;
	}

	for (int i = 0; i < deviceCount; i++) {
		CudaSafeCall(cudaSetDevice(i));
		cudaDeviceProp deviceProp;
		CudaSafeCall(cudaGetDeviceProperties(&deviceProp, i));
		printf("GPU #%d %s (%dx%d cores) (Cap %d.%d) (%.1f MB) (%s)\n",
			i, deviceProp.name, deviceProp.multiProcessorCount,
			_ConvertSMVer2Cores(deviceProp.major, deviceProp.minor),
			deviceProp.major, deviceProp.minor, (double)deviceProp.totalGlobalMem / 1048576.0,
			sComputeMode[deviceProp.computeMode]);
	}
}

// ----------------------------------------------------------------------------

GPUEngine::~GPUEngine()
{
	CudaSafeCall(cudaFree(inputKey[0]));
	CudaSafeCall(cudaFree(inputKey[1]));
	CudaSafeCall(cudaFree(inputHash));

	CudaSafeCall(cudaFreeHost(outputBufferPinned));
	CudaSafeCall(cudaFree(outputBuffer));

	CudaRandSafeCall(curandDestroyGenerator(prngGPU));
	CudaSafeCall(cudaStreamDestroy(rngStream));
	CudaSafeCall(cudaStreamDestroy(computeStream));

}

// ----------------------------------------------------------------------------

int GPUEngine::GetNbThread()
{
	return nbThread;
}

// ----------------------------------------------------------------------------

bool GPUEngine::CallKernel()
{

	// Reset nbFound and launch the kernel on the compute stream, consuming the
	// currently-ready key buffer.
	CudaSafeCall(cudaMemsetAsync(outputBuffer, 0, 4, computeStream));

	compute_hash<<<nbThread / nbThreadPerGroup, nbThreadPerGroup, 0, computeStream>>>
        (inputKey[keyBuf], inputHash, numHash160, maxFound, outputBuffer, startRange, endRange);

	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess) {
		printf("GPUEngine: callKernel: %s\n", cudaGetErrorString(err));
		return false;
	}
	return true;

}

// ----------------------------------------------------------------------------

bool GPUEngine::Step(std::vector<ITEM>& dataFound, bool spinWait)
{
	dataFound.clear();
	bool ret = true;

	// Launch the kernel on the current key buffer...
	ret = CallKernel();

	// ...and, concurrently, generate the keys for the next Step into the other
	// buffer on the RNG stream. This overlaps cuRAND with the compute kernel.
	Randomize(keyBuf ^ 1);

	// Wait for the kernel (and its outputBuffer reset) to complete.
	if (spinWait) {
		CudaSafeCall(cudaStreamSynchronize(computeStream));
	}
	else {
		// Poll to keep the CPU mostly idle while the kernel runs.
		while (cudaStreamQuery(computeStream) == cudaErrorNotReady) {
			Timer::SleepMillis(1);
		}
		CudaSafeCall(cudaStreamQuery(computeStream));
	}

	// Look for found
	CudaSafeCall(cudaMemcpy(outputBufferPinned, outputBuffer, 4, cudaMemcpyDeviceToHost));
	uint32_t nbFound = outputBufferPinned[0];
	if (nbFound > maxFound) {
		nbFound = maxFound;
	}

	// When can perform a standard copy, the kernel is eneded
	CudaSafeCall(cudaMemcpy(outputBufferPinned, outputBuffer, nbFound * ITEM_SIZE_A + 4, cudaMemcpyDeviceToHost));

	for (uint32_t i = 0; i < nbFound; i++) {
		uint32_t* itemPtr = outputBufferPinned + (i * ITEM_SIZE_A32 + 1);
		ITEM it;
		it.thId = itemPtr[0];
		it.pubKey = (uint8_t*)(itemPtr + 1);
		it.hash160 = (uint8_t*)(itemPtr + 10);
		dataFound.push_back(it);
	}

	// Make sure the next buffer's keys are fully generated, then swap to it.
	CudaSafeCall(cudaStreamSynchronize(rngStream));
	keyBuf ^= 1;

	return ret;
}

// ----------------------------------------------------------------------------

bool GPUEngine::Randomize(int buf)
{
	// Async on rngStream; the caller synchronizes rngStream before using buf.
	CudaRandSafeCall(curandGenerateLongLong(prngGPU, (unsigned long long*)inputKey[buf], nbThread * 4));

	return true;
}

// ----------------------------------------------------------------------------
