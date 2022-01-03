#include <array>

namespace ss
{
	namespace detail
	{
		template<class _T> using Invoke = typename _T::type;

		template<size_t...> struct seq { using type = seq; };

		template<class _S1, class _S2> struct concat;

		template<size_t... _i1, size_t... _i2>
		struct concat<seq<_i1...>, seq<_i2...>>
			: seq<_i1..., (sizeof...(_i1) + _i2)...> {};

		template<class _S1, class _S2>
		using Concat = Invoke<concat<_S1, _S2>>;

		template<size_t _n> struct gen_seq;
		template<size_t _n> using GenSeq = Invoke<gen_seq<_n>>;

		template<size_t _n>
		struct gen_seq : Concat<GenSeq<_n / 2>, GenSeq<_n - _n / 2>> {};

		template<> struct gen_seq<0> : seq<> {};
		template<> struct gen_seq<1> : seq<0> {};
	}

	template<size_t n>
	class static_str
	{
		std::array<char, n + 1> arr;

		template<size_t m, size_t... i1, size_t... i2>
		constexpr static_str(
			const static_str<m>& s1,
			const static_str<n - m>& s2,
			detail::seq<i1...>, detail::seq<i2...>
		)
			: arr{ s1[i1]..., s2[i2]..., 0 }
		{

		}

	public:
		constexpr static_str(std::array<char, n + 1> _arr) : arr{ _arr }
		{
		}

		template<size_t m>
		constexpr static_str(const static_str<m>& s1, const static_str<n - m>& s2)
			: static_str{ s1, s2, detail::gen_seq<m>{}, detail::gen_seq<n - m>{} }
		{

		}

		constexpr char operator[](size_t i) const
		{
			return arr[i];
		}

		constexpr size_t size() const
		{
			return n;
		}

		constexpr const char* c_str() const
		{
			return arr.data();
		}
	};

	namespace detail
	{
		template<size_t... digits>
		struct to_chars
		{
			static constexpr static_str<sizeof...(digits)> value = { std::array<char, sizeof...(digits) + 1>{ ('0' + digits)..., 0 } };
		};

		template<size_t rem, size_t... digits>
		struct explode : explode<rem / 10, rem % 10, digits...> {};

		template<size_t... digits>
		struct explode<0, digits...> : to_chars<digits...> {};
	}

	template<size_t num>
	struct num_to_string : detail::explode<num> {};

	template <size_t n1, size_t n2>
	constexpr static_str<n1 + n2> operator+(const static_str<n1>& s1, const static_str<n2>& s2)
	{
		return { s1, s2 };
	}

	namespace detail
	{
		template<size_t n, size_t... i>
		constexpr static_str<n - 1> from_literal(const char(&lit)[n], seq<i...>)
		{
			return { std::array<char, n>{ lit[i]... } };
		}
	}

	template<size_t n>
	constexpr static_str<n - 1> from_literal(const char(&lit)[n])
	{
		return detail::from_literal(lit, detail::gen_seq<n>{});
	}
}