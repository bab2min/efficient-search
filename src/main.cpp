#include <tuple>
#include <vector>
#include <string>
#include <unordered_set>
#include <random>
#include <limits>
#include <chrono>
#include <iostream>
#include <algorithm>
#include <numeric>

#include "balanced_binary.hpp"
#include "bst.hpp"

using namespace std;

template<class IntTy>
vector<IntTy> unique_rand_array(size_t size, bool uniform = true, size_t seed = 42)
{
	using SIntTy = typename conditional<sizeof(IntTy) == 1, int16_t, IntTy>::type;
	vector<IntTy> ret(size);
	mt19937_64 rng{ seed };

	unordered_set<IntTy> uniq;
	size_t i = 0;

	if (uniform)
	{
		uniform_int_distribution<SIntTy> dist{ numeric_limits<IntTy>::min(), numeric_limits<IntTy>::max() };
		while(i < size)
		{
			auto t = (IntTy)dist(rng);
			if (uniq.count(t)) continue;
			uniq.emplace(t);
			ret[i++] = t;
		}
	}
	else
	{
		uniform_int_distribution<SIntTy> dist{ numeric_limits<IntTy>::min() / 2, numeric_limits<IntTy>::max() / 2 + numeric_limits<IntTy>::max() / 4 };
		while (i < size)
		{
			auto t = (IntTy)(dist(rng) + dist(rng));
			if (uniq.count(t)) continue;
			uniq.emplace(t);
			ret[i++] = t;
		}
	}
	return ret;
}

template<class IntTy>
vector<IntTy> rand_array(size_t size, bool uniform = true, size_t seed = 42)
{
	using SIntTy = typename conditional<sizeof(IntTy) == 1, int16_t, IntTy>::type;
	vector<IntTy> ret(size);
	mt19937_64 rng{ seed };

	if (uniform)
	{
		uniform_int_distribution<SIntTy> dist{ numeric_limits<IntTy>::min(), numeric_limits<IntTy>::max() };
		for(auto& r : ret) r = (IntTy)dist(rng);
	}
	else
	{
		uniform_int_distribution<SIntTy> dist{ numeric_limits<IntTy>::min() / 2, numeric_limits<IntTy>::max() / 2 + numeric_limits<IntTy>::max() / 4 };
		for (auto& r : ret) r = (IntTy)(dist(rng) + dist(rng));
	}
	return ret;
}


struct ReferenceSearcher
{
	static constexpr const char* _name = "Reference";

	template<class KeyTy, class ValueTy>
	void prepare(KeyTy* keys, ValueTy* values, size_t size)
	{
		vector<size_t> idx(size);
		iota(idx.begin(), idx.end(), 0);

		sort(idx.begin(), idx.end(), [&](size_t a, size_t b)
		{
			return keys[a] < keys[b];
		});

		vector<KeyTy> temp_keys{ keys, keys + size };
		vector<ValueTy> temp_values{ values, values + size };

		for (size_t i = 0; i < size; ++i)
		{
			keys[i] = temp_keys[idx[i]];
			values[i] = temp_values[idx[i]];
		}
	}

	template<class KeyTy, class ValueTy>
	bool search(const KeyTy* keys, const ValueTy* values, size_t size, KeyTy target, ValueTy& found)
	{
		auto it = lower_bound(keys, keys + size, target);

		if (it == keys + size || *it != target) return false;

		found = values[it - keys];
		return true;
	}
};

struct BalancedBinarySearcher : public ReferenceSearcher
{
	static constexpr const char* _name = "BalancedBinary";

	template<class KeyTy, class ValueTy>
	bool search(const KeyTy* keys, const ValueTy* values, size_t size, KeyTy target, ValueTy& found)
	{
		size_t idx;
		if (!balanced_binary_search<false>(keys, size, target, idx)) return false;
		found = values[idx];
		return true;
	}
};

struct BalancedBinaryPrefetchSearcher : public ReferenceSearcher
{
	static constexpr const char* _name = "BalancedBinaryPrefetch";

	template<class KeyTy, class ValueTy>
	bool search(const KeyTy* keys, const ValueTy* values, size_t size, KeyTy target, ValueTy& found)
	{
		size_t idx;
		if (!balanced_binary_search<true>(keys, size, target, idx)) return false;
		found = values[idx];
		return true;
	}
};

struct BSTSearcher
{
	static constexpr const char* _name = "BinarySearchTree";

	template<class KeyTy, class ValueTy>
	void prepare(KeyTy* keys, ValueTy* values, size_t size)
	{
		vector<size_t> idx = bst_order(keys, size);

		vector<KeyTy> temp_keys{ keys, keys + size };
		vector<ValueTy> temp_values{ values, values + size };

		for (size_t i = 0; i < size; ++i)
		{
			keys[i] = temp_keys[idx[i]];
			values[i] = temp_values[idx[i]];
		}
	}

	template<class KeyTy, class ValueTy>
	bool search(const KeyTy* keys, const ValueTy* values, size_t size, KeyTy target, ValueTy& found)
	{
		size_t idx;
		if (!bst_search(keys, size, target, idx)) return false;
		found = values[idx];
		return true;
	}
};

struct N3STSearcher
{
	static constexpr const char* _name = "3-ary SearchTree";

	template<class KeyTy, class ValueTy>
	void prepare(KeyTy* keys, ValueTy* values, size_t size)
	{
		vector<size_t> idx = nst_order<3>(keys, size);

		vector<KeyTy> temp_keys{ keys, keys + size };
		vector<ValueTy> temp_values{ values, values + size };

		for (size_t i = 0; i < size; ++i)
		{
			keys[i] = temp_keys[idx[i]];
			values[i] = temp_values[idx[i]];
		}
	}

	template<class KeyTy, class ValueTy>
	bool search(const KeyTy* keys, const ValueTy* values, size_t size, KeyTy target, ValueTy& found)
	{
		size_t idx;
		if (!nst_search<3>(keys, size, target, idx)) return false;
		found = values[idx];
		return true;
	}
};

struct N5STSearcher
{
	static constexpr const char* _name = "5-ary SearchTree";

	template<class KeyTy, class ValueTy>
	void prepare(KeyTy* keys, ValueTy* values, size_t size)
	{
		vector<size_t> idx = nst_order<5>(keys, size);

		vector<KeyTy> temp_keys{ keys, keys + size };
		vector<ValueTy> temp_values{ values, values + size };

		for (size_t i = 0; i < size; ++i)
		{
			keys[i] = temp_keys[idx[i]];
			values[i] = temp_values[idx[i]];
		}
	}

	template<class KeyTy, class ValueTy>
	bool search(const KeyTy* keys, const ValueTy* values, size_t size, KeyTy target, ValueTy& found)
	{
		size_t idx;
		if (!nst_search<5>(keys, size, target, idx)) return false;
		found = values[idx];
		return true;
	}
};

#if defined(__SSE2__) || defined(__AVX2__)
struct SSE2STSearcher
{
	static constexpr const char* _name = "SSE2 SearchTree";

	template<class KeyTy, class ValueTy>
	void prepare(KeyTy* keys, ValueTy* values, size_t size)
	{
		static constexpr size_t n = 16 / sizeof(KeyTy) + 1;
		vector<size_t> idx = nst_order<n>(keys, size);

		vector<KeyTy> temp_keys{ keys, keys + size };
		vector<ValueTy> temp_values{ values, values + size };

		for (size_t i = 0; i < size; ++i)
		{
			keys[i] = temp_keys[idx[i]];
			values[i] = temp_values[idx[i]];
		}
	}

	template<class KeyTy, class ValueTy>
	bool search(const KeyTy* keys, const ValueTy* values, size_t size, KeyTy target, ValueTy& found)
	{
		static constexpr size_t n = 16 / sizeof(KeyTy) + 1;
		size_t idx;
		if (!nst_search_sse2<n>(keys, size, target, idx)) return false;
		found = values[idx];
		return true;
	}
};

struct SSE2ST2Searcher : public SSE2STSearcher
{
	static constexpr const char* _name = "SSE2 SearchTree (type2)";

	template<class KeyTy, class ValueTy>
	bool search(const KeyTy* keys, const ValueTy* values, size_t size, KeyTy target, ValueTy& found)
	{
		static constexpr size_t n = 16 / sizeof(KeyTy) + 1;
		size_t idx;
		if (!nst2_search_sse2<n>(keys, size, target, idx)) return false;
		found = values[idx];
		return true;
	}
};
#endif


#if defined(__AVX2__)
struct AVX2STSearcher
{
	static constexpr const char* _name = "AVX2 SearchTree";

	template<class KeyTy, class ValueTy>
	void prepare(KeyTy* keys, ValueTy* values, size_t size)
	{
		static constexpr size_t n = 32 / sizeof(KeyTy) + 1;
		vector<size_t> idx = nst_order<n>(keys, size);

		vector<KeyTy> temp_keys{ keys, keys + size };
		vector<ValueTy> temp_values{ values, values + size };

		for (size_t i = 0; i < size; ++i)
		{
			keys[i] = temp_keys[idx[i]];
			values[i] = temp_values[idx[i]];
		}
	}

	template<class KeyTy, class ValueTy>
	bool search(const KeyTy* keys, const ValueTy* values, size_t size, KeyTy target, ValueTy& found)
	{
		static constexpr size_t n = 32 / sizeof(KeyTy) + 1;
		size_t idx;
		if (!nst_search_avx2<n>(keys, size, target, idx)) return false;
		found = values[idx];
		return true;
	}
};

struct AVX2ST2Searcher : public AVX2STSearcher
{
	static constexpr const char* _name = "AVX2 SearchTree (type2)";

	template<class KeyTy, class ValueTy>
	bool search(const KeyTy* keys, const ValueTy* values, size_t size, KeyTy target, ValueTy& found)
	{
		static constexpr size_t n = 32 / sizeof(KeyTy) + 1;
		size_t idx;
		if (!nst2_search_avx2<n>(keys, size, target, idx)) return false;
		found = values[idx];
		return true;
	}
};
#endif

template<class KeyTy, class Searcher>
pair<vector<size_t>, double> benchmark(Searcher&& searcher, size_t size, bool uniform, size_t sample_size, double hit_rate = 0.5)
{
	auto keys = unique_rand_array<KeyTy>(size, uniform);
	auto values = vector<size_t>(size);
	iota(values.begin(), values.end(), 0);

	const size_t target_size = 8192;
	auto targets = rand_array<KeyTy>(target_size, true, 777);
	for (size_t i = (size_t)(target_size * hit_rate); i < target_size; ++i)
	{
		targets[i] = keys[(size_t)targets[i] % size];
	}
	shuffle(targets.begin(), targets.end(), mt19937_64{});

	auto results = vector<size_t>(target_size, size);

	searcher.prepare(keys.data(), values.data(), size);

	chrono::high_resolution_clock::time_point start_time = chrono::high_resolution_clock::now();

	for (size_t i = 0; i < sample_size; ++i)
	{
		searcher.search(keys.data(), values.data(), size, targets[i % target_size], results[i % target_size]);
	}
	chrono::high_resolution_clock::time_point end_time = chrono::high_resolution_clock::now();
	double elapsed = chrono::duration<double, std::milli>{ end_time - start_time }.count();
	return make_pair(move(results), elapsed);
}

template<class KeyTy>
void run_benchmark_partial(const vector<size_t>& ref, double* accum, double* accum_sq, size_t size, bool uniform, size_t sample_size)
{
}

template<class KeyTy, class First, class... Rest>
void run_benchmark_partial(const vector<size_t>& ref, double* accum, double* accum_sq, size_t size, bool uniform, size_t sample_size)
{
	auto r = benchmark<KeyTy>(First{}, size, uniform, sample_size);
	if (ref != r.first)
	{
		printf("    %s yields a wrong result!\n", First::_name);
	}
	*accum += r.second;
	*accum_sq += r.second * r.second;
	run_benchmark_partial<KeyTy, Rest...>(ref, accum + 1, accum_sq + 1, size, uniform, sample_size);
}


template<class KeyTy, class... Searchers>
void run_benchmark_set(tuple<Searchers...>, size_t size, bool uniform, size_t sample_size, size_t repeat = 10)
{
	vector<double> accum(sizeof ... (Searchers) + 1), accum_sq(sizeof ... (Searchers) + 1);
	for (size_t i = 0; i < repeat; ++i)
	{
		auto ref_result = benchmark<KeyTy>(ReferenceSearcher{}, size, uniform, sample_size);
		accum[0] += ref_result.second;
		accum_sq[0] += ref_result.second * ref_result.second;
		run_benchmark_partial<KeyTy, Searchers...>(ref_result.first, accum.data() + 1, accum_sq.data() + 1, size, uniform, sample_size);
	}

	static const char* names[] = {
		ReferenceSearcher::_name,
		(Searchers::_name)...,
	};

	for (size_t i = 0; i < accum.size(); ++i)
	{
		double mean = accum[i] / repeat;
		double stdev = sqrt(max((accum_sq[i] / repeat) - mean * mean, 0.));
		printf("  %-25s: %9.5g ms (%5.3g ms)\n", names[i], mean, stdev);
	}
}

int main()
{
	const size_t sample_size = 1000 * 1000;
	const size_t repeat = 20;

	using Searchers = tuple<
		BalancedBinarySearcher,
		BalancedBinaryPrefetchSearcher,
		N3STSearcher,
		N5STSearcher,
#if defined(__SSE2__) || defined(__AVX2__)
		SSE2STSearcher,
		SSE2ST2Searcher,
#endif
#ifdef __AVX2__
		AVX2STSearcher,
		AVX2ST2Searcher,
#endif
		BSTSearcher
	>;
	
	for (bool uniform : {true, false})
	{
		for (size_t size : { 25, 50, 100, 200, 400, 800, 1600, 3200, 6400, 12800 })
		{
			printf("======== int16_t, size=%zd, uniform_dist=%s ========\n", size, uniform ? "true" : "false");
			run_benchmark_set<int16_t>(Searchers{}, size, uniform, sample_size, repeat);
			printf("\n\n");
		}

		for (size_t size : { 25, 50, 100, 200, 400, 800, 1600, 3200, 6400, 12800 })
		{
			printf("======== int32_t, size=%zd, uniform_dist=%s ========\n", size, uniform ? "true" : "false");
			run_benchmark_set<int32_t>(Searchers{}, size, uniform, sample_size, repeat);
			printf("\n\n");
		}
	}
	return 0;
}
