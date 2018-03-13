
#ifndef DENSE_LSH_HPP
#define DENSE_LSH_HPP

#include <algorithm>
#include <vector>
#include <utility>
#include <string>
#include <tuple>
#include <memory>
#include <cmath>
#include <unordered_map>
#include <set>

#include "aux.hpp"
#include "thread_aux.hpp"
#include "layers.hpp"
#include "hash_functor.hpp"

template <typename Opt, typename Weight = double>
struct DenseLSH : public Layer_2D<Weight> {
private:
    using Matrix = std::vector<std::vector<Weight>>;
    using TableRowElem = 
        std::pair<size_t, std::shared_ptr<std::vector<Weight>>>;
    using TableRow = std::vector<TableRowElem>;
    using Table = std::unordered_map<size_t, TableRow>;
    using ActiveSet = std::vector<TableRowElem>;


    Opt optimizer;


    // a tables hashes are stored by index. table[i]'s hashes are
    // in table_hashes[i].

    size_t append_length; // size to append for query and PP functions.
    size_t num_hash_per_table; 
    size_t num_tables;

    std::vector<Table> tables;
    std::vector<std::vector<HashFunctor<Weight>>> hashes;

    // contains the active set. Saved so it can be
    // used in the backward pass.
    TableRow sub_matr;


    std::vector<Weight> query(std::vector<Weight> q) {
        std::vector<Weight> append(append_length, 0.5);
        q.insert(q.end(), append.begin(), append.end());
        return q;
    }

    std::vector<Weight> preprocess(std::vector<Weight> x) {
        std::vector<Weight> append(append_length, 0); 
        for(size_t i = 0; i < append_length; ++i) {
            append[i] = std::pow(std::sqrt(aux::dot(x, x)), 2*(i+1));
        }
        x.insert(x.end(), append.begin(), append.end());
        return x;
    }


    void fill_table(size_t table) {
        Table& t = tables[table];    
        std::vector<HashFunctor<Weight>>& table_hash = hashes[table];

        
        Matrix weights_trans = aux::transpose(this->weights);

        for(size_t r = 0; r < weights_trans.size(); ++r) {
            for(auto hash : table_hash) {
                size_t table_index = hash(preprocess(weights_trans[r]));
                
                auto ins_ptr = 
                    std::make_shared<std::vector<Weight>>(weights_trans[r]);

                auto ins = std::make_pair(r, ins_ptr);

                t[table_index].push_back(ins);
            }
        }
    }

    std::set<size_t> 
    apply_hashes(size_t table, const std::vector<Weight>& row) {
        // apply all of a tables hashes to one row.
        std::set<size_t> indices;
        for(auto hash : hashes[table]) {
            indices.insert(hash(query(row)));
        }
        return indices;
    }

    ActiveSet generate_active_set(const Matrix& input) {
        ActiveSet active_set;         

        for(size_t r = 0; r < input.size(); ++r) {
            for(size_t t = 0; t < tables.size(); ++t) {
                std::set<size_t> indices = apply_hashes(t, input[r]);
                for(auto iter = indices.begin();
                    iter != indices.end();
                    ++iter) 
                {
                    TableRow ins = tables[t][*iter];
                    active_set.insert(active_set.end(),
                                      ins.begin(), ins.end());
                    if(active_set.size() >= this->size) {
                        return ActiveSet(active_set.begin(), 
                                         active_set.begin() + this->size);
                    }
                }
            }
        }
        return active_set;
    }

public:

    DenseLSH():Layer_2D<Weight>("dense_lsh") {}
    DenseLSH(int num_nodes, int input_size, size_t m, Weight r, 
             size_t num_buckets, size_t K, size_t L, double learning_rate):
        Layer_2D<Weight>(num_nodes, input_size, "dense_lsh", learning_rate),
        optimizer(),
        append_length(m),
        num_hash_per_table(K),
        num_tables(L),
        tables(L),
        hashes(K, std::vector<HashFunctor<Weight>>(
                        num_hash_per_table,
                        HashFunctor<Weight>(input_size, num_buckets, r))),
     
        sub_matr() 
    {
        std::cout << "L: " << L << '\n';
        std::cout << this->hashes.size() << '\n';
        // some form of uniform xavier intialization.
        double v = std::sqrt(6.0 / input_size);
        for(auto& row : this->weights) {
            for(auto& item : row) {
                item = aux::gen_double(-0.5 * v, 0.5 * v);
            }
        }
        for(size_t t = 0; t < tables.size(); ++t) {
            fill_table(t);
        }
    }

    ~DenseLSH() = default;

    DenseLSH(DenseLSH&& other): 
        Layer_2D<Weight>(std::move(other)),
        append_length(std::move(other.append_length)),
        num_hash_per_table(std::move(other.num_hash_per_table)),
        num_tables(std::move(other.num_tables)),
        tables(std::move(other.tables)),
        hashes(std::move(other.hashes))
    {}

    DenseLSH(const DenseLSH& other): 
        Layer_2D<Weight>(other),
        append_length(other.append_length),
        num_hash_per_table(other.num_hash_per_table),
        num_tables(other.num_tables),
        tables(other.tables),
        hashes(other.hashes)
    {}

    DenseLSH* clone() {
        return new DenseLSH(*this);
    }

    size_t forward_count = 0;
    Matrix forward_pass(const Matrix& input) {
        // TODO
        this->last_input = input;

        sub_matr = generate_active_set(input);
        
        Matrix matr(sub_matr.size());
        for(size_t r = 0; r < matr.size(); ++r) {
            matr[r] = *sub_matr[r].second;
        }

        Matrix matr_trans = aux::transpose(matr);

        Matrix out_matr = aux::matmul(input, matr_trans);
        

        this->last_output = out_matr;
        // apply bias to out_matr

        return out_matr;
    }

    Matrix async_forward_pass(const Matrix& input, size_t n_threads) {    
        // TODO
        return input;
    }

    Matrix operator()(const Matrix& input) {
        return forward_pass(input);
    }

    Matrix backward_pass(const Matrix& d_out) {
        // TODO
        
        Matrix matr(sub_matr.size());
        for(size_t r = 0; r < matr.size(); ++r) {
            matr[r] = *sub_matr[r].second;
        }



        return d_out;
    }

    Matrix async_backward_pass(const Matrix& d_out, size_t n_threads) {
        // TODO
        return d_out;
    }

    // some helper functions.
    size_t layer_size() const {
        return this->size;
    }

    Matrix read_weights() const {
        return this->weight;
    }
};

#endif
