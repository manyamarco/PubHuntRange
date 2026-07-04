/*
 * This file is part of the PubHunt distribution (https://github.com/kanhavishva/PubHunt).
 * Copyright (c) 2021 KV.
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

#include "PubHunt.h"
#include "IntGroup.h"
#include "Timer.h"
#include <cstring>
#include <cmath>
#include <algorithm>
#include <iostream>
#include <cassert>
#ifndef WIN64
#include <pthread.h>
#endif

using namespace std;

// ----------------------------------------------------------------------------

PubHunt::PubHunt(const std::vector<std::vector<uint8_t>>& inputHashes, const std::string& outputFile, uint64_t startRange, uint64_t endRange,
    const std::string& checkpointFile, uint32_t checkpointInterval)
{
    this->outputFile = outputFile;
    this->nbGPUThread = 0;
    this->maxFound = 65536;
    this->startRange = startRange;
    this->endRange = endRange;

    this->checkpointFile = checkpointFile;
    this->checkpointInterval = checkpointInterval;
    this->resumeCount = 0;
    this->resumeTime = 0.0;

	this->numHash160 = inputHashes.size();

	this->hash160 = new uint32_t[5 * numHash160];

	uint8_t* hash160_t = (uint8_t*)hash160;

	for (int i = 0; i < numHash160; i++) {
		auto h = inputHashes.at(i);
		for (int j = 0; j < h.size(); j++) {
			hash160_t[i * 20 + j] = h.at(j);
		}
	}

}

// ----------------------------------------------------------------------------

PubHunt::~PubHunt()
{
	delete[] hash160;
}

// ----------------------------------------------------------------------------

double log1(double x)
{
	// Use taylor series to approximate log(1-x)
	return -x - (x * x) / 2.0 - (x * x * x) / 3.0 - (x * x * x * x) / 4.0;
}

// ----------------------------------------------------------------------------

void PubHunt::output(const ITEM& result)
{

#ifdef WIN64
	WaitForSingleObject(ghMutex, INFINITE);
#else
	pthread_mutex_lock(&ghMutex);
#endif

	FILE* f = stdout;
	bool needToClose = false;

	if (outputFile.length() > 0) {
		f = fopen(outputFile.c_str(), "a");
		if (f == NULL) {
			printf("Cannot open %s for writing\n", outputFile.c_str());
			f = stdout;
		}
		else {
			needToClose = true;
		}
	}

	if (!needToClose)
		printf("\n");

	// write pubkey
	fprintf(f, "PubKey : ");
	for (int32_t i = 0; i < 36; i++) {
		if (!(i == 1 || i == 2 || i == 3))
			fprintf(f, "%02hhx", result.pubKey[i]);
	}
	fprintf(f, "\n");

	// write hash160
	fprintf(f, "Hash160: ");
	for (int32_t i = 0; i < 20; i++) {
		fprintf(f, "%02hhx", result.hash160[i]);
	}
	fprintf(f, "\n");

	fprintf(f, "=================================================================================\n");

	if (needToClose)
		fclose(f);

#ifdef WIN64
	ReleaseMutex(ghMutex);
#else
	pthread_mutex_unlock(&ghMutex);
#endif

}

// ----------------------------------------------------------------------------

#ifdef WIN64
DWORD WINAPI _FindKeyGPU(LPVOID lpParam)
{
#else
void* _FindKeyGPU(void* lpParam)
{
#endif
	TH_PARAM* p = (TH_PARAM*)lpParam;
	p->obj->FindKeyGPU(p);
	return 0;
}

// ----------------------------------------------------------------------------

void PubHunt::FindKeyGPU(TH_PARAM * ph)
{

	bool ok = true;

#ifdef WITHGPU

	// Global init
	int thId = ph->threadId;

	GPUEngine* g = new GPUEngine(ph->gridSizeX, ph->gridSizeY, ph->gpuId, maxFound, hash160, numHash160, startRange, endRange, resumeCount);

	int nbThread = g->GetNbThread();

	vector<ITEM> found;

	printf("GPU          : %s\n", g->deviceName.c_str());

	counters[thId] = 0;

	ph->hasStarted = true;

	// GPU Thread
	while (ok && !endOfSearch) {

		ok = g->Step(found, false);

		for (int i = 0; i < (int)found.size() && !endOfSearch; i++) {
			ITEM it = found[i];
			output(it);
			nbFoundKey++;
		}

		counters[thId] += nbThread * 2;
	}

	delete g;

#else
	ph->hasStarted = true;
	printf("GPU code not compiled, use -DWITHGPU when compiling.\n");
#endif

	ph->isRunning = false;

}

// ----------------------------------------------------------------------------

bool PubHunt::isAlive(TH_PARAM * p)
{

	bool isAlive = true;
	int total = nbGPUThread;
	for (int i = 0; i < total; i++)
		isAlive = isAlive && p[i].isRunning;

	return isAlive;

}

// ----------------------------------------------------------------------------

bool PubHunt::hasStarted(TH_PARAM * p)
{

	bool hasStarted = true;
	int total = nbGPUThread;
	for (int i = 0; i < total; i++)
		hasStarted = hasStarted && p[i].hasStarted;

	return hasStarted;

}

// ----------------------------------------------------------------------------

uint64_t PubHunt::getGPUCount()
{

	uint64_t count = 0;
	for (int i = 0; i < nbGPUThread; i++)
		count += counters[0x80L + i];
	return count;

}

// ----------------------------------------------------------------------------

// 64-bit FNV-1a digest of the input hash160 set. Used to make sure a checkpoint
// is only resumed when it belongs to the same search (same target hashes).
uint64_t PubHunt::inputDigest()
{
	uint64_t h = 1469598103934665603ULL; // FNV offset basis
	const uint8_t* p = (const uint8_t*)hash160;
	size_t n = (size_t)numHash160 * 20;
	for (size_t i = 0; i < n; i++) {
		h ^= (uint64_t)p[i];
		h *= 1099511628211ULL; // FNV prime
	}
	return h;
}

// ----------------------------------------------------------------------------

// Persist cumulative progress atomically (write to a temp file then rename)
// so an interrupted write can never corrupt an existing checkpoint.
bool PubHunt::SaveCheckpoint(uint64_t totalKeys, double elapsed)
{
	if (checkpointFile.empty())
		return false;

	std::string tmp = checkpointFile + ".tmp";
	FILE* f = fopen(tmp.c_str(), "w");
	if (f == NULL)
		return false;

	fprintf(f, "PUBHUNT_CHECKPOINT_V1\n");
	fprintf(f, "startRange=%016llx\n", (unsigned long long)startRange);
	fprintf(f, "endRange=%016llx\n", (unsigned long long)endRange);
	fprintf(f, "numHash160=%d\n", numHash160);
	fprintf(f, "inputDigest=%016llx\n", (unsigned long long)inputDigest());
	fprintf(f, "totalKeys=%llu\n", (unsigned long long)totalKeys);
	fprintf(f, "elapsedSec=%.3f\n", elapsed);
	fprintf(f, "foundKeys=%d\n", nbFoundKey);
	fflush(f);
	fclose(f);

	// Atomic replace. std::rename fails on Windows if the target exists, so
	// remove it first; the .tmp file still holds a valid checkpoint meanwhile.
	remove(checkpointFile.c_str());
	if (rename(tmp.c_str(), checkpointFile.c_str()) != 0) {
		remove(tmp.c_str());
		return false;
	}
	return true;
}

// ----------------------------------------------------------------------------

// Load a checkpoint, but only accept it if it matches the current search
// parameters. Returns false (fresh start) on any mismatch or parse error.
bool PubHunt::LoadCheckpoint(uint64_t& totalKeys, double& elapsed, int& foundKeys)
{
	if (checkpointFile.empty())
		return false;

	FILE* f = fopen(checkpointFile.c_str(), "r");
	if (f == NULL)
		return false;

	char magic[64] = { 0 };
	uint64_t cpStart = 0, cpEnd = 0, cpDigest = 0, cpKeys = 0;
	int cpNum = -1, cpFound = 0;
	double cpElapsed = 0.0;

	bool ok = (fscanf(f, "%63s\n", magic) == 1) && (strcmp(magic, "PUBHUNT_CHECKPOINT_V1") == 0);
	ok = ok && (fscanf(f, "startRange=%llx\n", (unsigned long long*)&cpStart) == 1);
	ok = ok && (fscanf(f, "endRange=%llx\n", (unsigned long long*)&cpEnd) == 1);
	ok = ok && (fscanf(f, "numHash160=%d\n", &cpNum) == 1);
	ok = ok && (fscanf(f, "inputDigest=%llx\n", (unsigned long long*)&cpDigest) == 1);
	ok = ok && (fscanf(f, "totalKeys=%llu\n", (unsigned long long*)&cpKeys) == 1);
	ok = ok && (fscanf(f, "elapsedSec=%lf\n", &cpElapsed) == 1);
	ok = ok && (fscanf(f, "foundKeys=%d\n", &cpFound) == 1);
	fclose(f);

	if (!ok) {
		printf("Checkpoint: ignoring '%s' (unreadable or wrong format)\n", checkpointFile.c_str());
		return false;
	}

	if (cpStart != startRange || cpEnd != endRange || cpNum != numHash160 || cpDigest != inputDigest()) {
		printf("Checkpoint: ignoring '%s' (parameters differ from current search)\n", checkpointFile.c_str());
		return false;
	}

	totalKeys = cpKeys;
	elapsed = cpElapsed;
	foundKeys = cpFound;
	return true;
}

// ----------------------------------------------------------------------------

void PubHunt::Search(std::vector<int> gpuId, std::vector<int> gridSize, bool& should_exit)
{

	double t0;
	double t1;
	endOfSearch = false;

	nbGPUThread = (int)gpuId.size();
	nbFoundKey = 0;

	memset(counters, 0, sizeof(counters));

	// Resume from a previous run if a matching checkpoint exists. This must run
	// before the GPU threads start so resumeCount is used as the RNG offset.
	resumeCount = 0;
	resumeTime = 0.0;
	{
		uint64_t cpKeys = 0; double cpElapsed = 0.0; int cpFound = 0;
		if (LoadCheckpoint(cpKeys, cpElapsed, cpFound)) {
			resumeCount = cpKeys;
			resumeTime = cpElapsed;
			nbFoundKey = cpFound;
			char tstr[256] = { 0 };
			printf("Checkpoint: resuming from '%s' [T: %s scanned, %d found]\n",
				checkpointFile.c_str(), toTimeStr((int)resumeTime, tstr), nbFoundKey);
		}
	}

	TH_PARAM* params = (TH_PARAM*)malloc((nbGPUThread) * sizeof(TH_PARAM));
	memset(params, 0, (nbGPUThread) * sizeof(TH_PARAM));

	// Launch GPU threads
	for (int i = 0; i < nbGPUThread; i++) {
		params[i].obj = this;
		params[i].threadId = 0x80L + i;
		params[i].isRunning = true;
		params[i].gpuId = gpuId[i];
		params[i].gridSizeX = gridSize[2 * i];
		params[i].gridSizeY = gridSize[2 * i + 1];

#ifdef WIN64
		DWORD thread_id;
		CreateThread(NULL, 0, _FindKeyGPU, (void*)(params + (i)), 0, &thread_id);
#else
		pthread_t thread_id;
		pthread_create(&thread_id, NULL, &_FindKeyGPU, (void*)(params + (i)));
#endif
	}

#ifndef WIN64
	setvbuf(stdout, NULL, _IONBF, 0);
#endif

	uint64_t lastCount = resumeCount;
	uint64_t gpuCount = 0;
	uint64_t lastGPUCount = 0;

	// Key rate smoothing filter
#define FILTER_SIZE 8
	double lastkeyRate[FILTER_SIZE];
	double lastGpukeyRate[FILTER_SIZE];
	uint32_t filterPos = 0;

	double keyRate = 0.0;
	double gpuKeyRate = 0.0;
	char timeStr[256];

	memset(lastkeyRate, 0, sizeof(lastkeyRate));
	memset(lastGpukeyRate, 0, sizeof(lastkeyRate));

	// Wait that all threads have started
	while (!hasStarted(params)) {
		Timer::SleepMillis(500);
	}
	printf("\n");

	// Reset timer
	Timer::Init();
	t0 = Timer::get_tick();
	startTime = t0;

	Int ICount;

	double lastCheckpoint = t0;

	while (isAlive(params)) {

		int delay = 2000;
		while (isAlive(params) && delay > 0) {
			Timer::SleepMillis(500);
			delay -= 500;
		}

		gpuCount = getGPUCount();
		// count = cumulative keys including any resumed from a checkpoint.
		uint64_t count = gpuCount + resumeCount;
		ICount.SetInt64(count);
		int completedBits = ICount.GetBitLength();

		t1 = Timer::get_tick();
		keyRate = (double)(count - lastCount) / (t1 - t0);
		gpuKeyRate = (double)(gpuCount - lastGPUCount) / (t1 - t0);
		lastkeyRate[filterPos % FILTER_SIZE] = keyRate;
		lastGpukeyRate[filterPos % FILTER_SIZE] = gpuKeyRate;
		filterPos++;

		// KeyRate smoothing
		double avgGpuKeyRate = 0.0;
		uint32_t nbSample;
		for (nbSample = 0; (nbSample < FILTER_SIZE) && (nbSample < filterPos); nbSample++) {
			avgGpuKeyRate += lastGpukeyRate[nbSample];
		}
		avgGpuKeyRate /= (double)(nbSample);

		if (isAlive(params)) {
			memset(timeStr, '\0', 256);
			printf("\r[%s] [GPU: %.2f MK/s] [T: %s (%d bit)] [F: %d]  ",
				toTimeStr((int)(t1 + resumeTime), timeStr),
				avgGpuKeyRate / 1000000.0,
				formatThousands(count).c_str(),
				completedBits,
				nbFoundKey);
		}

		// Periodic checkpoint so an unexpected termination loses at most
		// `checkpointInterval` seconds of accounted progress.
		if (checkpointInterval > 0 && (t1 - lastCheckpoint) >= (double)checkpointInterval) {
			SaveCheckpoint(count, t1 + resumeTime);
			lastCheckpoint = t1;
		}

		lastCount = count;
		lastGPUCount = gpuCount;
		t0 = t1;
		if (should_exit)
			endOfSearch = true;
	}

	// Final checkpoint on a clean exit (Ctrl-C or a thread stopping) so the
	// next launch resumes from the latest progress.
	if (checkpointInterval > 0) {
		SaveCheckpoint(getGPUCount() + resumeCount, Timer::get_tick() + resumeTime);
	}

	free(params);

}

// ----------------------------------------------------------------------------

std::string PubHunt::formatThousands(uint64_t x)
{
	char buf[32] = "";

	sprintf(buf, "%llu", x);

	std::string s(buf);

	int len = (int)s.length();

	int numCommas = (len - 1) / 3;

	if (numCommas == 0) {
		return s;
	}

	std::string result = "";

	int count = ((len % 3) == 0) ? 0 : (3 - (len % 3));

	for (int i = 0; i < len; i++) {
		result += s[i];

		if (count++ == 2 && i < len - 1) {
			result += ",";
			count = 0;
		}
	}
	return result;
}

char* PubHunt::toTimeStr(int sec, char* timeStr)
{
	int h, m, s;
	h = (sec / 3600);
	m = (sec - (3600 * h)) / 60;
	s = (sec - (3600 * h) - (m * 60));
	sprintf(timeStr, "%0*d:%0*d:%0*d", 2, h, 2, m, 2, s);
	return (char*)timeStr;
}

// ----------------------------------------------------------------------------
