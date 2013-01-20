#pragma once

#include <vector>
#include <algorithm>

template <typename T>
T median(std::vector<T> v)
{
    int med_idx = floor(v.size() / 2);
    nth_element(v.begin(), v.begin()+med_idx, v.end());
    return v[med_idx];
}

std::vector<int> indexShuffle(int begin,int end);