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
