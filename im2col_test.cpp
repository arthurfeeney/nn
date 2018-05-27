
#include <vector>
#include <iostream>

#include "include/im2col.hpp"




int main() {
    using Row = std::vector<int>;
    using Matrix = std::vector<Row>;
    using Image = std::vector<Matrix>;

    auto im = Image(10, Matrix(10, Row(10, 1)));

    auto matr = im2col::im_2_col<Image, Matrix>(im, 3, 1, 8, 8);

    
    for(auto& r : matr) {
        for(auto& val : r) {
            std::cout << val << ' ';
        }
        std::cout << '\n';
    }


    return 0;
}
