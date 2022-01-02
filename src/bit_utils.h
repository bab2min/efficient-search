#pragma once

#include <cstdint>
#ifdef _MSC_VER
#include <intrin.h>
#pragma intrinsic(__popcnt)
#pragma intrinsic(_BitScanForward)
#pragma intrinsic(_BitScanForward64)
#pragma intrinsic(_BitScanReverse)
#pragma intrinsic(_BitScanReverse64)
#endif

#if _WIN32 || _M_X64 || __x86_64__
#include <xmmintrin.h>
#include <immintrin.h>
#endif

inline int count_trailing_zeroes(uint32_t v)
{
	if (v == 0)
	{
		return 32;
	}
#if defined(__GNUC__)
	return __builtin_ctz(v);
#elif defined(_MSC_VER)
	unsigned long count;
	_BitScanForward(&count, v);
	return (int)count;
#else
	// See Stanford bithacks, count the consecutive zero bits (trailing) on the
	// right with multiply and lookup:
	// http://graphics.stanford.edu/~seander/bithacks.html#ZerosOnRightMultLookup
	static const uint8_t tbl[32] = { 0,  1,  28, 2,  29, 14, 24, 3,  30, 22, 20,
									15, 25, 17, 4,  8,  31, 27, 13, 23, 21, 19,
									16, 7,  26, 12, 18, 6,  11, 5,  10, 9 };
	return (int)tbl[((uint32_t)((v & -v) * 0x077CB531U)) >> 27];
#endif
}

inline int count_trailing_zeroes(uint64_t v)
{
	if (v == 0)
	{
		return 64;
	}
#if defined(__GNUC__)
	return __builtin_ctzll(v);
#elif defined(_MSC_VER) && defined(_M_X64)
	unsigned long count;
	_BitScanForward64(&count, v);
	return (int)count;
#else
	return (uint32_t)v ? count_trailing_zeroes((uint32_t)v)
		: 32 + count_trailing_zeroes((uint32_t)(v >> 32));
#endif
}

inline int count_leading_zeroes(uint32_t v)
{
	if (v == 0)
	{
		return 32;
	}
#if defined(__GNUC__)
	return __builtin_clz(v);
#elif defined(_MSC_VER)
	unsigned long count;
	_BitScanReverse(&count, v);
	// BitScanReverse gives the bit position (0 for the LSB, then 1, etc.) of the
	// first bit that is 1, when looking from the MSB. To count leading zeros, we
	// need to adjust that.
	return 31 - int(count);
#else
	// See Stanford bithacks, find the log base 2 of an N-bit integer in
	// O(lg(N)) operations with multiply and lookup:
	// http://graphics.stanford.edu/~seander/bithacks.html#IntegerLogDeBruijn
	static const uint8_t tbl[32] = { 31, 22, 30, 21, 18, 10, 29, 2,  20, 17, 15,
									13, 9,  6,  28, 1,  23, 19, 11, 3,  16, 14,
									7,  24, 12, 4,  8,  25, 5,  26, 27, 0 };
	v = v | (v >> 1);
	v = v | (v >> 2);
	v = v | (v >> 4);
	v = v | (v >> 8);
	v = v | (v >> 16);
	return (int)tbl[((uint32_t)(v * 0x07C4ACDDU)) >> 27];
#endif
}

inline int count_leading_zeroes(uint64_t v)
{
	if (v == 0)
	{
		return 64;
	}
#if defined(__GNUC__)
	return __builtin_clzll(v);
#elif defined(_MSC_VER) && defined(_M_X64)
	unsigned long count;
	_BitScanReverse64(&count, v);
	return 63 - int(count);
#else
	return v >> 32 ? count_leading_zeroes((uint32_t)(v >> 32))
		: 32 + count_leading_zeroes((uint32_t)v);
#endif
}

inline int ceil_log2(uint32_t v) { return 32 - count_leading_zeroes(v - 1); }

inline int ceil_log2(uint64_t v) { return 64 - count_leading_zeroes(v - 1); }

#ifdef __APPLE__
inline int ceil_log2(size_t v) { return ceil_log2((uint64_t)v); }
#endif

inline uint32_t popcount(uint32_t v)
{
#if defined(__GNUC__)
	return __builtin_popcount(v);
#elif defined(_MSC_VER)
	return __popcnt(v);
#else
	throw "";
#endif
}
