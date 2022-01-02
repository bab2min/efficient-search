#pragma once
#include "bit_utils.h"

inline void prefetch(const void* ptr)
{
#if _WIN32 || _M_X64 || __x86_64__
	_mm_prefetch((const char*)ptr, _MM_HINT_T0);
#elif defined(__GNUC__)
	__builtin_prefetch(ptr);
#endif
}

template<bool use_prefetch, class IntTy>
bool balanced_binary_search(const IntTy* keys, size_t size, IntTy target, size_t& ret)
{
	static constexpr size_t cacheLineSize = 64 / sizeof(IntTy);

	int height = ceil_log2(size + 1);
	size_t dist = (size_t)1 << (size_t)(height - 1);
	size_t mid = size - dist;
	dist >>= 1;
	size_t left1 = 0, left2 = mid + 1;
	while (height-- > 0)
	{
		if (use_prefetch && dist >= cacheLineSize / sizeof(IntTy))
		{
			prefetch(&keys[left1 + dist - 1]);
			prefetch(&keys[left2 + dist - 1]);
		}
		if (target > keys[mid]) left1 = left2;
		left2 = left1 + dist;
		mid = left1 + dist - 1;
		dist >>= 1;
	}
	if (left1 == size || keys[left1] != target) return false;
	ret = left1;
	return true;
}

static constexpr int get_bit_size(size_t n)
{
	return n <= 1 ? 0 : (get_bit_size(n >> 1) + 1);
}

#if defined(__SSE2__) || defined(__AVX2__)
template<bool use_prefetch, class IntTy>
bool balanced_binary_search_sse2(const IntTy* keys, size_t size, IntTy target, size_t& ret)
{
	static constexpr size_t cacheLineSize = 64 / sizeof(IntTy);
	static constexpr int minH = get_bit_size(16 / sizeof(IntTy));
	static constexpr int realMinH = minH > 2 ? minH : 0;

	int height = ceil_log2(size + 1);
	size_t dist = (size_t)1 << (size_t)(height - 1);
	size_t mid = size - dist;
	dist >>= 1;
	size_t left1 = 0, left2 = mid + 1;
	while (height-- > realMinH)
	{
		if (use_prefetch && dist >= cacheLineSize / sizeof(IntTy))
		{
			prefetch(&keys[left1 + dist - 1]);
			prefetch(&keys[left2 + dist - 1]);
		}
		if (target > keys[mid]) left1 = left2;
		left2 = left1 + dist;
		mid = left1 + dist - 1;
		dist >>= 1;
	}
	
	uint32_t mask;
	__m128i ptarget;
	switch(sizeof(IntTy))
	{
	case 1:
		ptarget = _mm_set1_epi8((int8_t)target);
		mask = _mm_movemask_epi8(_mm_cmpeq_epi8(
			_mm_loadu_si128((const __m128i*)&keys[left1]),
			ptarget
		));
		break;
	case 2:
		ptarget = _mm_set1_epi16((int16_t)target);
		mask = _mm_movemask_epi8(_mm_cmpeq_epi16(
			_mm_loadu_si128((const __m128i*)&keys[left1]),
			ptarget
		));
		break;
	case 4:
		ptarget = _mm_set1_epi32((int32_t)target);
		mask = _mm_movemask_epi8(_mm_cmpeq_epi32(
			_mm_loadu_si128((const __m128i*)&keys[left1]),
			ptarget
		));
		break;
	}

	size_t i = count_trailing_zeroes(mask);
	if (mask && left1 + (i / sizeof(IntTy)) < size)
	{
		ret = left1 + (i / sizeof(IntTy));
		return true;
	}
	return false;
}
#endif

#ifdef __AVX2__
template<bool use_prefetch, class IntTy>
bool balanced_binary_search_avx2(const IntTy* keys, size_t size, IntTy target, size_t& ret)
{
	static constexpr size_t cacheLineSize = 64 / sizeof(IntTy);
	static constexpr int minH = get_bit_size(32 / sizeof(IntTy));
	static constexpr int realMinH = minH > 2 ? minH : 0;

	int height = ceil_log2(size + 1);
	size_t dist = (size_t)1 << (size_t)(height - 1);
	size_t mid = size - dist;
	dist >>= 1;
	size_t left1 = 0, left2 = mid + 1;
	while (height-- > realMinH)
	{
		if (use_prefetch && dist >= cacheLineSize / sizeof(IntTy))
		{
			prefetch(&keys[left1 + dist - 1]);
			prefetch(&keys[left2 + dist - 1]);
		}
		if (target > keys[mid]) left1 = left2;
		left2 = left1 + dist;
		mid = left1 + dist - 1;
		dist >>= 1;
	}

	uint32_t mask;
	__m256i ptarget;
	switch (sizeof(IntTy))
	{
	case 1:
		ptarget = _mm256_set1_epi8((int8_t)target);
		mask = _mm256_movemask_epi8(_mm256_cmpeq_epi8(
			_mm256_loadu_si256((const __m256i*)&keys[left1]),
			ptarget
		));
		break;
	case 2:
		ptarget = _mm256_set1_epi16((int16_t)target);
		mask = _mm256_movemask_epi8(_mm256_cmpeq_epi16(
			_mm256_loadu_si256((const __m256i*)&keys[left1]),
			ptarget
		));
		break;
	case 4:
		ptarget = _mm256_set1_epi32((int32_t)target);
		mask = _mm256_movemask_epi8(_mm256_cmpeq_epi32(
			_mm256_loadu_si256((const __m256i*)&keys[left1]),
			ptarget
		));
		break;
	}

	size_t i = count_trailing_zeroes(mask);
	if (mask && left1 + (i / sizeof(IntTy)) < size)
	{
		ret = left1 + (i / sizeof(IntTy));
		return true;
	}
	return false;
}
#endif

#if defined(__ARM_NEON__) || defined(__ARM_NEON)

#endif
