
#ifndef OPERATION_GATE_HPP
#define OPERATION_GATE_HPP

#include <string>
#include <vector>

/*
 * An operation gate applies a simple operation to the output of two layers.
 * 
 *
 */


template<typename OpFunc, typename Weight = double>
class OperationGate : public Layer_2D<Weight> {
private:
    OpFunc op;

public:
    OperationGate(OpFunc op): op(op)
    {} 


  

}

#endif
