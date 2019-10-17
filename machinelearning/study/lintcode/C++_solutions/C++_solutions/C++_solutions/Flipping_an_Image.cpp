#include <iostream>
#include <algorithm>

class Solution {
public:
	std::vector<std::vector<int>> flipAndInvertImage(std::vector<std::vector<int>>& A) {
		for (auto& row : A) {
			int i = 0;
			int j = row.size();
			while (i != j && i != --j) {
				std::swap(row[i], row[j]);
				++i;
			}

			for (auto& c : row) {
				c = (c == 1) ? 0 : 1;
			}
		}
		return A
	}
};