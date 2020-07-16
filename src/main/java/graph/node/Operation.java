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

package graph.node;

import math.Tensor;

import java.util.Arrays;

/**
 * An <code>Operation</code> represents a graph.node which applies an operation to one or more inputs. <code>Operation</code> implements
 * <code>Callable</code>, as their results must be accessed as a <code>Future</code> list when called by an <code>ExecutorService</code>
 * to ensure that consumers have the required values to proceed before running. The <code>Operation</code> graph.node must also implement a
 * differentiation method, which calculates its derivative.
 */
public abstract class Operation extends Node {

    /**
     * Constructs an <code>Operation</code> graph.node, adding itself to the consumers of its children.
     *
     * @param children the children of the graph.node
     * @see Node#Node(Node...)
     */
    protected Operation(Node... children) {
        super(children);

        for (Node child : children) {
            child.getConsumers().add(this);
        }
    }

    /**
     * Computes the output of the graph.node as a tensor, given the inputs of the graph.node.
     *
     * @param inputs the inputs of this graph.node, as tensors
     * @return the output of this graph.node, as a tensor
     */
    protected abstract Tensor computeOutput(Tensor[] inputs);

    /**
     * Calls the method, which computes the output of the graph.node, inputting the outputs of children
     * nodes.
     *
     * @return the output of this graph.node, as a tensor
     */
    @Override
    protected Tensor computeOutput() {
        return computeOutput(Arrays.stream(children).map(Results::getOutput).toArray(Tensor[]::new));
    }
}
