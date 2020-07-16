/*
 * MIT License
 *
 * Copyright (c) 2020 Patrick Song
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

package graph.node.operation;

import graph.node.Node;
import graph.node.Operation;
import graph.node.Results;
import math.Tensor;

import java.util.Map;

/**
 * A <code>Multiplication</code> graph.node represents a graph.node which applies an <b>element-wise</b>
 * multiplication operation to multiple tensors.
 */
public class Multiplication extends Operation {

    public Multiplication(Node... children) {
        super(children);
    }

    @Override protected Map<Long, Tensor> computeGradients(Map<Long, Tensor> gradients, Tensor delta) {
        for (Node child : children) {
            Tensor gradient = Tensor.unbroadcast(
                    Operations.multiplication(delta, Operations.division(Results.getOutput(this), Results.getOutput(child))),
                    Results.getOutput(child).getDimensions());

            gradients.put(child.getID(), gradient);
        }

        return gradients;
    }

    /**
     * Element-wise multiplies the inputted tensors, broadcasting them if necessary.
     *
     * @param inputs the inputs of this graph.node, as tensors
     * @return the output of this graph.node, as a tensor
     */
    @Override protected Tensor computeOutput(Tensor[] inputs) {
        return Operations.multiplication(inputs);
    }
}
