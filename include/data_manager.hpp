#include <vector>
#include <random>
#include <utility>
#include <algorithm>

template<typename DataCont, typename LabelCont, typename Weight = double>
class Data_Manager {
public:
    Data_Manager(DataCont train_data,
                 LabelCont train_labels,
                 DataCont test_data,
                 LabelCont test_labels,
                 size_t batch_size,
                 size_t ensemble_size):
        train_data(train_data),
        train_labels(train_labels),
        test_data(test_data),
        test_labels(test_labels),
        data_order(train_labels.size()),
        batch_size(batch_size),
        ensemble_size(ensemble_size),
        trs(train_labels.size()),
        tes(test_labels.size())
    {
        std::iota(data_order.begin(), data_order.end(), 0);
    }

    std::pair<DataCont, LabelCont> get_batch() {
        if(batch_index + batch_size >= data_order.size()) {
            batch_index = 0;
        }

        std::vector<size_t> indices(data_order.begin() + batch_index,
                                    data_order.begin() + batch_index + batch_size);

        DataCont data_batch(batch_size);
        LabelCont label_batch(batch_size);

        for(int k = 0; k < batch_size; ++k) {
            data_batch[k] = train_data[indices[k]];
            label_batch[k] = train_labels[indices[k]];
        }

        batch_index += batch_size;

        return std::make_pair(data_batch, label_batch);
    }

    std::vector<std::pair<DataCont, LabelCont>>
    chunk_batch(const std::pair<DataCont, LabelCont>& batch) {
        DataCont data = batch.first;
        LabelCont labels = batch.second;
        size_t chunk_size = batch_size / ensemble_size;
        std::vector<std::pair<DataCont, LabelCont>> chunks(ensemble_size);
        for(size_t i = 0, index = 0;
            i < batch_size;
            i += chunk_size, ++index) 
        {
            DataCont d_c(data.begin() + i, data.begin() + i + chunk_size);
            LabelCont l_c(labels.begin() + i, labels.begin() + i + chunk_size);
            chunks[index] = std::make_pair(d_c, l_c);
        }
        return chunks;
    }

    void process_all_data() {
        chunkified_batches.clear();
        for(int index = 0; index < train_size(); index += batch_size) {
            chunkified_batches.push_back(chunk_batch(get_batch()));
        }
    }  

    void shuffle_data() {
        std::random_device rd;
        std::mt19937 g(rd());
        std::shuffle(data_order.begin(), data_order.begin(), g);
    }

    std::pair<DataCont, LabelCont> get_chunk(size_t net, size_t chunk) {
        return chunkified_batches[chunk][net];
    }

    size_t train_size() {
        return trs;
    }

    size_t test_size() {
        return tes;
    }

    std::pair<DataCont, LabelCont> test() {
        return std::make_pair(test_data, test_labels);
    }

    size_t step_size() {
        return batch_size;
    }

    size_t num_train_batches() {
        return chunkified_batches.size();
    }

private:
    DataCont train_data;
    LabelCont train_labels;
    DataCont test_data;
    LabelCont test_labels;

    std::vector<std::vector<std::pair<DataCont, LabelCont>>> chunkified_batches;

    std::vector<size_t> data_order;
    size_t batch_size;
    size_t batch_index = 0;
    size_t ensemble_size;

    size_t trs; // size of the train set
    size_t tes; // size of the test set
};
