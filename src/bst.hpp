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
		for (size_t k = 0; k < n - 1; ++k)
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
using CondBool = typename std::enable_if<(n - 1) * sizeof(IntTy) == p, bool>::type;

#if defined(__SSE2__) || defined(__AVX2__)
#include <xmmintrin.h>

template<size_t n, class IntTy>
CondBool<n, IntTy, 16> nst_search_sse2(const IntTy* keys, size_t size, IntTy target, size_t& ret)
{
	size_t i = 0;
	
	__m128i ptarget;
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
		__m128i pkey = _mm_loadu_si128((const __m128i*)&keys[i]);
		__m128i peq, pgt;
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
		pkey = _mm256_loadu_si256((const __m256i*) & keys[i]);
		peq, pgt;
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
#endif