#include "Utils.h"

std::vector<int> indexShuffle(int begin,int end){
    std::vector<int> indexes(end-begin);
    for (int i=begin,j=0; i<end; i++,j++){
        indexes[j]=i;
    }
    std::random_shuffle(indexes.begin(),indexes.end());
    return indexes;
}
