#include "LSH.h"
#include <stddef.h>
#include <thread>
#include <iostream>

LSH::LSH(int K_, int L_, int T_) : K(K_), L(L_), THREADS(T_)
{
	for(int idx = 0; idx < L; ++idx)
	{
		std::unordered_map<int, std::unordered_set<int>> table;
		tables.emplace_back(std::move(table));
	}
}

// Insert a single element into LSH hash tables
void LSH::insert(const int* fp, const int item_id)
{
	for(int idx = 0; idx < L; ++idx)
	{
		const int key = fp[idx];
		std::unordered_map<int, std::unordered_set<int>>& table = tables[idx];
		if(table.find(key) == table.end())
		{
			std::unordered_set<int> value;
			table.emplace(key, std::move(value));
		}
		table[key].emplace(item_id);
	}
}

// Insert a single element into LSH hash tables
void LSH::erase(const int* fp, const int item_id)
{
	for(int idx = 0; idx < L; ++idx)
	{
		tables[idx][fp[idx]].erase(item_id);
	}
}

std::unordered_set<int> LSH::query(const int* fp, const size_t N, int* cL)
{
	std::unordered_set<int> result;
	for(*cL = 0; *cL < L && result.size() <= N; *cL += 1)
	{
                const int table_idx = *cL;
                const int bucket_idx = fp[table_idx];

		std::unordered_map<int, std::unordered_set<int>>& table = tables[table_idx];
		if(table.find(bucket_idx) != table.end())
		{
			const std::unordered_set<int>& bucket = table[bucket_idx];
			result.insert(bucket.begin(), bucket.end());
		}
	}
	return result;
}

void LSH::clear()
{
	for(int idx = 0; idx < L; ++idx)
	{
		tables[idx].clear();
	}

}
