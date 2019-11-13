/*
 * MIT License
 *
 * Copyright (c) 2019 Patrick Song
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

package neural.graph.node.operation;

import neural.graph.node.Node;
import neural.math.Tensor;

/**
 * A <code>Multiplication</code> node represents a node which applies an <b>element-wise</b>
 * multiplication operation to multiple tensors.
 */
public class Multiplication extends Operation {

    public Multiplication(Node... children) {
        super(children);
    }

    /**
     * Element-wise multiplies the inputted tensors, broadcasting them if necessary.
     *
     * @param inputs the inputs of this node, as tensors
     * @return the output of this node, as a tensor
     */
    @Override public Tensor computeOutput(Tensor[] inputs) {
        // check if the tensors require broadcasting, then broadcast if necessary
        if (Tensor.isDimensionsMismatch(inputs)) {
            inputs = Tensor.broadcast(inputs);
        }

        // create an output tensor with the output dimensions
        Tensor result = Tensor.zeros(inputs[0].getDimensions());
        for (int i = 0; i < result.getLength(); i++) {
            float product = 1;

            // add each tensor element-wise
            // because they are broadcast, this may be done by absolute index
            for (Tensor input : inputs) {
                product *= input.get(i);
            }

            // set the proper value in the result
            result.set(product, i);
        }

        return result;
    }
}
