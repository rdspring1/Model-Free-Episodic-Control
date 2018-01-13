#include <vector>
#include <unordered_map>
#include <unordered_set>

class LSH
{
	private:
		// Members
		const int K;
		const int L;
		const int THREADS;
		std::vector<std::unordered_map<int, std::unordered_set<int>>> tables;

	public:
		LSH(int, int, int);
		void insert(const int*, const int);
		std::unordered_set<int> query(const int* fp, const size_t N, int* cL);
		void erase(const int*, const int);
		void clear();
};
