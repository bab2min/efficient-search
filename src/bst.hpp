#pragma once

#include <vector>
#include <algorithm>
#include <numeric>

#include "bit_utils.h"

template<class KeyTy>
std::vector<size_t> bst_order(const KeyTy* keys, size_t size)
{
	std::vector<size_t> ret(size), idx(size);
	std::iota(idx.begin(), idx.end(), 0);

	std::sort(idx.begin(), idx.end(), [&](size_t a, size_t b)
	{
		return keys[a] < keys[b];
	});

	size_t height = 0;
	for (size_t s = size; s > 0; s >>= 1) height++;

	size_t complete_size = (1 << height) - 1;
	size_t off = complete_size - size;
	size_t off_start = complete_size + 1 - off * 2;

	size_t i = 0;
	for (size_t h = 0; h < height; ++h)
	{
		size_t start = (1 << (height - h - 1)) - 1;
		size_t stride = 1 << (height - h);
		for (size_t r = start; r < complete_size; r += stride)
		{
			size_t f = r;
			if (f > off_start) f -= (f - off_start + 1) / 2;
			ret[i++] = idx[f];
			if (i >= size) break;
		}
	}

	return ret;
}

template<class KeyTy>
bool bst_search(const KeyTy* keys, size_t size, KeyTy target, size_t& ret)
{
	size_t i = 0;
	while (i < size)
	{
		if (target == keys[i])
		{
			ret = i;
			return true;
		}
		else if (target < keys[i])
		{
			i = i * 2 + 1;
		}
		else
		{
			i = i * 2 + 2;
		}
	}
	return false;
}

template<class Ty>
Ty powi(Ty a, size_t b)
{
	if (b == 0) return 1;
	if (b == 1) return a;
	if (b == 2) return a * a;
	if (b == 3) return a * a * a;

	return powi(a, b / 2) * powi(a, b - (b / 2));
}

template<size_t n, class KeyTy>
std::vector<size_t> nst_order(const KeyTy* keys, size_t size)
{
	std::vector<size_t> ret(size), idx(size);
	std::iota(idx.begin(), idx.end(), 0);

	std::sort(idx.begin(), idx.end(), [&](size_t a, size_t b)
	{
		return keys[a] < keys[b];
	});

	size_t height = 0;
	for (size_t s = size; s > 0; s /= n) height++;

	size_t complete_size = powi(n, height) - 1;
	size_t off = complete_size - size;
	size_t off_start = complete_size + 1 - ((off + n - 2) / (n - 1) + off);

	size_t i = 0;
	for (size_t h = 0; h < height; ++h)
	{
		size_t stride = powi(n, height - h - 1);
		size_t start = stride - 1;
		
		for (size_t r = start; r < complete_size; r += stride)
		{
			for (size_t k = 0; k < n - 1; ++k, r += stride)
			{
				size_t f = r;
				if (f > off_start) f -= (f - off_start) - (f - off_start) / n;
				ret[i++] = idx[f];
				if (i >= size) break;
			}
			if (i >= size) break;
		}
	}
	return ret;
}

template<size_t n, class KeyTy>
bool nst_search(const KeyTy* keys, size_t size, KeyTy target, size_t& ret)
{
	size_t i = 0;
	while (i < size)
	{
		size_t r = 0;
		size_t ke = std::min(n - 1, size - i);
		for (size_t k = 0; k < ke; ++k)
		{
			if (target == keys[i + k])
			{
				ret = i + k;
				return true;
			}
			if (target > keys[i + k]) r++;
		}

		i = i * n + (n - 1) * (r + 1);
	}
	return false;
}


template<size_t n, class IntTy, size_t p>
using CondBool = typename std::enable_if<(n - 1) % (p / sizeof(IntTy)) == 0, bool>::type;

#if defined(__SSE2__) || defined(__AVX2__)
#include <xmmintrin.h>

template<size_t n, class IntTy>
CondBool<n, IntTy, 16> nst_search_sse2(const IntTy* keys, size_t size, IntTy target, size_t& ret)
{
	static constexpr size_t packet_size = 16 / sizeof(IntTy);
	static constexpr size_t inner_size = (n - 1) / packet_size;
	size_t i = 0;

	__m128i ptarget, pkey, peq, pgt;
	switch (sizeof(IntTy))
	{
	case 1:
		ptarget = _mm_set1_epi8(target);
		break;
	case 2:
		ptarget = _mm_set1_epi16(target);
		break;
	case 4:
		ptarget = _mm_set1_epi32(target);
		break;
	}

	while (i < size)
	{
		pkey = _mm_loadu_si128((const __m128i*)&keys[i]);
		switch (sizeof(IntTy))
		{
		case 1:
			peq = _mm_cmpeq_epi8(ptarget, pkey);
			pgt = _mm_cmpgt_epi8(ptarget, pkey);
			break;
		case 2:
			peq = _mm_cmpeq_epi16(ptarget, pkey);
			pgt = _mm_cmpgt_epi16(ptarget, pkey);
			break;
		case 4:
			peq = _mm_cmpeq_epi32(ptarget, pkey);
			pgt = _mm_cmpgt_epi32(ptarget, pkey);
			break;
		}

		uint32_t m = _mm_movemask_epi8(peq);
		uint32_t p = count_trailing_zeroes(m);
		if (m && (i + p / sizeof(IntTy)) < size)
		{
			ret = i + p / sizeof(IntTy);
			return true;
		}
		size_t r = popcount(_mm_movemask_epi8(pgt)) / sizeof(IntTy);
		i = i * n + (n - 1) * (r + 1);
	}
	return false;
}

template<size_t n, class IntTy>
CondBool<n, IntTy, 16> nst2_search_sse2(const IntTy* keys, size_t size, IntTy target, size_t& ret)
{
	size_t i = 0;

	__m128i ptarget, pkey, pgt;
	switch (sizeof(IntTy))
	{
	case 1:
		ptarget = _mm_set1_epi8(target);
		break;
	case 2:
		ptarget = _mm_set1_epi16(target);
		break;
	case 4:
		ptarget = _mm_set1_epi32(target);
		break;
	}

	while (i + 16 / sizeof(IntTy) < size)
	{
		pkey = _mm_loadu_si128((const __m128i*)&keys[i]);
		switch (sizeof(IntTy))
		{
		case 1:
			pgt = _mm_cmpgt_epi8(ptarget, pkey);
			break;
		case 2:
			pgt = _mm_cmpgt_epi16(ptarget, pkey);
			break;
		case 4:
			pgt = _mm_cmpgt_epi32(ptarget, pkey);
			break;
		}

		size_t r = popcount(_mm_movemask_epi8(pgt)) / sizeof(IntTy);
		if (keys[i + r] == target)
		{
			ret = i + r;
			return true;
		}

		i = i * n + (n - 1) * (r + 1);
	}

	if (i < size)
	{
		pkey = _mm_loadu_si128((const __m128i*)&keys[i]);
		switch (sizeof(IntTy))
		{
		case 1:
			pgt = _mm_cmpeq_epi8(ptarget, pkey);
			break;
		case 2:
			pgt = _mm_cmpeq_epi16(ptarget, pkey);
			break;
		case 4:
			pgt = _mm_cmpeq_epi32(ptarget, pkey);
			break;
		}

		uint32_t m = _mm_movemask_epi8(pgt);
		uint32_t p = count_trailing_zeroes(m);
		if (m && (i + p / sizeof(IntTy)) < size)
		{
			ret = i + p / sizeof(IntTy);
			return true;
		}
	}
	return false;
}
#endif

#ifdef __AVX2__
#include <immintrin.h>

template<size_t n, class IntTy>
CondBool<n, IntTy, 32> nst_search_avx2(const IntTy* keys, size_t size, IntTy target, size_t& ret)
{
	size_t i = 0;

	__m256i ptarget, pkey, peq, pgt;
	switch (sizeof(IntTy))
	{
	case 1:
		ptarget = _mm256_set1_epi8(target);
		break;
	case 2:
		ptarget = _mm256_set1_epi16(target);
		break;
	case 4:
		ptarget = _mm256_set1_epi32(target);
		break;
	}

	while (i < size)
	{
		pkey = _mm256_loadu_si256((const __m256i*)&keys[i]);
		switch (sizeof(IntTy))
		{
		case 1:
			peq = _mm256_cmpeq_epi8(ptarget, pkey);
			pgt = _mm256_cmpgt_epi8(ptarget, pkey);
			break;
		case 2:
			peq = _mm256_cmpeq_epi16(ptarget, pkey);
			pgt = _mm256_cmpgt_epi16(ptarget, pkey);
			break;
		case 4:
			peq = _mm256_cmpeq_epi32(ptarget, pkey);
			pgt = _mm256_cmpgt_epi32(ptarget, pkey);
			break;
		}

		uint32_t m = _mm256_movemask_epi8(peq);
		uint32_t p = count_trailing_zeroes(m);
		if (m && (i + p / sizeof(IntTy)) < size)
		{
			ret = i + p / sizeof(IntTy);
			return true;
		}

		size_t r = popcount(_mm256_movemask_epi8(pgt)) / sizeof(IntTy);
		i = i * n + (n - 1) * (r + 1);
	}
	return false;
}

template<size_t n, class IntTy>
CondBool<n, IntTy, 32> nst2_search_avx2(const IntTy* keys, size_t size, IntTy target, size_t& ret)
{
	size_t i = 0;

	__m256i ptarget, pkey, peq, pgt;
	switch (sizeof(IntTy))
	{
	case 1:
		ptarget = _mm256_set1_epi8(target);
		break;
	case 2:
		ptarget = _mm256_set1_epi16(target);
		break;
	case 4:
		ptarget = _mm256_set1_epi32(target);
		break;
	}

	while (i + 32 / sizeof(IntTy) < size)
	{
		pkey = _mm256_loadu_si256((const __m256i*)&keys[i]);
		peq, pgt;
		switch (sizeof(IntTy))
		{
		case 1:
			pgt = _mm256_cmpgt_epi8(ptarget, pkey);
			break;
		case 2:
			pgt = _mm256_cmpgt_epi16(ptarget, pkey);
			break;
		case 4:
			pgt = _mm256_cmpgt_epi32(ptarget, pkey);
			break;
		}

		size_t r = popcount(_mm256_movemask_epi8(pgt)) / sizeof(IntTy);
		if (keys[i + r] == target)
		{
			ret = i + r;
			return true;
		}

		i = i * n + (n - 1) * (r + 1);
	}

	if (i < size)
	{
		pkey = _mm256_loadu_si256((const __m256i*)&keys[i]);
		switch (sizeof(IntTy))
		{
		case 1:
			pgt = _mm256_cmpeq_epi8(ptarget, pkey);
			break;
		case 2:
			pgt = _mm256_cmpeq_epi16(ptarget, pkey);
			break;
		case 4:
			pgt = _mm256_cmpeq_epi32(ptarget, pkey);
			break;
		}

		uint32_t m = _mm256_movemask_epi8(pgt);
		uint32_t p = count_trailing_zeroes(m);
		if (m && (i + p / sizeof(IntTy)) < size)
		{
			ret = i + p / sizeof(IntTy);
			return true;
		}
	}
	return false;
}
#endif

#if defined(__ARM_NEON__) || defined(__ARM_NEON)
#include <arm_neon.h>

inline uint32_t pop_unit_count(uint8x16_t val) {
	const uint8x16_t mask = { 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 };
	return vaddvq_u8(vandq_u8(val, mask));
}

inline uint32_t pop_unit_count(uint16x8_t val) {
	const uint16x8_t mask = { 1, 1, 1, 1, 1, 1, 1, 1 };
	return vaddvq_u16(vandq_u16(val, mask));
}

inline uint32_t pop_unit_count(uint32x4_t val) {
	const uint32x4_t mask = { 1, 1, 1, 1 };
	return vaddvq_u32(vandq_u32(val, mask));
}

inline bool neon_lookup(int8x16_t pkeys, int8x16_t ptarget, size_t size, size_t& ret)
{
	size_t found;
	static const uint8_t __attribute__((aligned(16))) idx[16][16] = {
		{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16 },
		{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 0 },
		{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 0, 0 },
		{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 0, 0, 0 },
		{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 0, 0, 0, 0 },
		{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 0, 0, 0, 0, 0 },
		{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 0, 0, 0, 0, 0, 0 },
		{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 0, 0, 0, 0, 0, 0 },
		{ 1, 2, 3, 4, 5, 6, 7, 8, 0, 0, 0, 0, 0, 0, 0, 0 },
		{ 1, 2, 3, 4, 5, 6, 7, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
		{ 1, 2, 3, 4, 5, 6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
		{ 1, 2, 3, 4, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
		{ 1, 2, 3, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
		{ 1, 2, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
		{ 1, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
		{ 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
	};
	uint8x16_t selected = vandq_u8(vceqq_s8(
		pkeys,
		ptarget
	), vld1q_u8(idx[16 - std::min(size, (size_t)16)]));
	found = vaddvq_u8(selected);
	
	if (found && found - 1 < size)
	{
		ret = found - 1;
		return true;
	}
	return false;
}

inline bool neon_lookup(int16x8_t pkeys, int16x8_t ptarget, size_t size, size_t& ret)
{
	size_t found;
	static const uint16_t __attribute__((aligned(16))) idx[8][8] = {
		{ 1, 2, 3, 4, 5, 6, 7, 8 },
		{ 1, 2, 3, 4, 5, 6, 7, 0 },
		{ 1, 2, 3, 4, 5, 6, 0, 0 },
		{ 1, 2, 3, 4, 5, 0, 0, 0 },
		{ 1, 2, 3, 4, 0, 0, 0, 0 },
		{ 1, 2, 3, 0, 0, 0, 0, 0 },
		{ 1, 2, 0, 0, 0, 0, 0, 0 },
		{ 1, 0, 0, 0, 0, 0, 0, 0 },
	};
	uint16x8_t selected = vandq_u16(vceqq_s16(
		pkeys,
		ptarget
	), vld1q_u16(idx[8 - std::min(size, (size_t)8)]));
	found = vaddvq_u16(selected);
	
	if (found && found - 1 < size)
	{
		ret = found - 1;
		return true;
	}
	return false;
}

inline bool neon_lookup(int32x4_t pkeys, int32x4_t ptarget, size_t size, size_t& ret)
{
	size_t found;
	static const uint32_t __attribute__((aligned(16))) idx[4][4] = {
		{ 1, 2, 3, 4 },
		{ 1, 2, 3, 0 },
		{ 1, 2, 0, 0 },
		{ 1, 0, 0, 0 },
	};
	uint32x4_t selected = vandq_u32(vceqq_s32(
		pkeys,
		ptarget
	), vld1q_u32(idx[4 - std::min(size, (size_t)4)]));
	found = vaddvq_u32(selected);

	if (found && found - 1 < size)
	{
		ret = found - 1;
		return true;
	}
	return false;
}

template<size_t n>
bool nst_search_neon(const int8_t* keys, size_t size, int8_t target, size_t& ret)
{
	size_t i = 0;

	int8x16_t ptarget, pkey;
	uint8x16_t pgt;
	ptarget = vdupq_n_s8(target);

	while (i < size)
	{
		pkey = vld1q_s8(&keys[i]);
		pgt = vcgtq_s8(ptarget, pkey);

		if (neon_lookup(pkey, ptarget, size - i, ret))
		{
			ret += i;
			return true;
		}

		size_t r = pop_unit_count(pgt);
		i = i * n + (n - 1) * (r + 1);
	}
	return false;
}

template<size_t n>
bool nst_search_neon(const int16_t* keys, size_t size, int16_t target, size_t& ret)
{
	size_t i = 0;

	int16x8_t ptarget, pkey;
	uint16x8_t pgt;
	ptarget = vdupq_n_s16(target);
	
	while (i < size)
	{
		pkey = vld1q_s16(&keys[i]);
		pgt = vcgtq_s16(ptarget, pkey);
	
		if (neon_lookup(pkey, ptarget, size - i, ret))
		{
			ret += i;
			return true;
		}

		size_t r = pop_unit_count(pgt);
		i = i * n + (n - 1) * (r + 1);
	}
	return false;
}

template<size_t n>
bool nst_search_neon(const int32_t* keys, size_t size, int32_t target, size_t& ret)
{
	size_t i = 0;

	int32x4_t ptarget, pkey;
	uint32x4_t pgt;
	ptarget = vdupq_n_s32(target);
	
	while (i < size)
	{
		pkey = vld1q_s32(&keys[i]);
		pgt = vcgtq_s32(ptarget, pkey);

		if (neon_lookup(pkey, ptarget, size - i, ret))
		{
			ret += i;
			return true;
		}

		size_t r = pop_unit_count(pgt);
		i = i * n + (n - 1) * (r + 1);
	}
	return false;
}

template<size_t n>
bool nst2_search_neon(const int8_t* keys, size_t size, int8_t target, size_t& ret)
{
	size_t i = 0;

	int8x16_t ptarget, pkey;
	uint8x16_t pgt;
	ptarget = vdupq_n_s8(target);

	while (i + 16 < size)
	{
		pkey = vld1q_s8(&keys[i]);
		pgt = vcgtq_s8(ptarget, pkey);

		size_t r = pop_unit_count(pgt);
		if (keys[i + r] == target)
		{
			ret = i + r;
			return true;
		}

		i = i * n + (n - 1) * (r + 1);
	}

	if (i < size)
	{
		pkey = vld1q_s8(&keys[i]);

		if (neon_lookup(pkey, ptarget, size - i, ret))
		{
			ret += i;
			return true;
		}
	}
	return false;
}

template<size_t n>
bool nst2_search_neon(const int16_t* keys, size_t size, int16_t target, size_t& ret)
{
	size_t i = 0;

	int16x8_t ptarget, pkey;
	uint16x8_t pgt;
	ptarget = vdupq_n_s16(target);

	while (i + 8 < size)
	{
		pkey = vld1q_s16(&keys[i]);
		pgt = vcgtq_s16(ptarget, pkey);

		size_t r = pop_unit_count(pgt);
		if (keys[i + r] == target)
		{
			ret = i + r;
			return true;
		}

		i = i * n + (n - 1) * (r + 1);
	}

	if (i < size)
	{
		pkey = vld1q_s16(&keys[i]);

		if (neon_lookup(pkey, ptarget, size - i, ret))
		{
			ret += i;
			return true;
		}
	}
	return false;
}

template<size_t n>
bool nst2_search_neon(const int32_t* keys, size_t size, int32_t target, size_t& ret)
{
	size_t i = 0;

	int32x4_t ptarget, pkey;
	uint32x4_t pgt;
	ptarget = vdupq_n_s32(target);

	while (i + 4 < size)
	{
		pkey = vld1q_s32(&keys[i]);
		pgt = vcgtq_s32(ptarget, pkey);

		size_t r = pop_unit_count(pgt);
		if (keys[i + r] == target)
		{
			ret = i + r;
			return true;
		}

		i = i * n + (n - 1) * (r + 1);
	}

	if (i < size)
	{
		pkey = vld1q_s32(&keys[i]);

		if (neon_lookup(pkey, ptarget, size - i, ret))
		{
			ret += i;
			return true;
		}
	}
	return false;
}

#endif
