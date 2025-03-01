#ifndef _COMMON_UTIL_H_
#define _COMMON_UTIL_H_

#if defined (_WIN64) || defined	(WIN32)
#include "windows.h"
#include <time.h>
__inline long long ntiGetTimeUsec()
{
	LARGE_INTEGER m_liPerfFreq = { 0 };
	LARGE_INTEGER m_liPerfStart = { 0 };
	QueryPerformanceFrequency(&m_liPerfFreq);
	QueryPerformanceCounter(&m_liPerfStart);
	return (long long)(m_liPerfStart.QuadPart * 1.0 * 1000 * 1000 / m_liPerfFreq.QuadPart);
}
#else
#include <sys/time.h>
__inline long long ntiGetTimeUsec()
{
	struct timeval t;
	gettimeofday(&t, 0);
	return (long long)((long long)t.tv_sec * 1000 * 1000 + t.tv_usec);
}
#endif	//defined (_WIN64) || defined (WIN32)


#define NTI_INIT_TIME()
		int s32IndexTime = 0;                                                               \
		long long s64StartTime = 0, s64EndTime = 0;                                         \
		float fTimeCountStatis = 0;                                                         \
		static float fCurModulTimeCountStatic[256] = {0.0};                                 \
		static float fAllTimeCount = 0;                                                     \
		static int s32Count_Time = 0;	                                                    \
		char strTimeName[256];


#define NTI_START_TIME()	                      				                            \
		s64StartTime = ntiGetTimeUsec();


#define NTI_END_TIME(...)					                                                \
		sprintf(strTimeName, __VA_ARGS__);                                                  \
		s64EndTime = ntiGetTimeUsec();						                                \
		fTimeCountStatis += (s64EndTime - s64StartTime) / 1000.0f;						    \
		fCurModulTimeCountStatic[s32IndexTime] += (s64EndTime - s64StartTime) / 1000.0f;    \
		if (s32IndexTime == 0) s32Count_Time++;                                             \
		printf("%-30s time: %8.3f ms, avg %5.3f ms\n",                                      \
			strTimeName, (s64EndTime - s64StartTime) / 1000.0f,                             \
			fCurModulTimeCountStatic[s32IndexTime] / s32Count_Time);                        \
		s32IndexTime++;

#endif  // _COMMON_UTIL_H_