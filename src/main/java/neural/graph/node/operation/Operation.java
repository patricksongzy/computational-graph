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
import neural.graph.node.Results;
import neural.math.Tensor;

import java.util.Arrays;
import java.util.concurrent.Callable;

/**
 * An <code>Operation</code> represents a node which applies an operation to one or more inputs. <code>Operation</code> implements
 * <code>Callable</code>, as their results must be accessed as a <code>Future</code> list when called by an <code>ExecutorService</code>
 * to ensure that consumers have the required values to proceed before running. The <code>Operation</code> node must also implement a
 * differentiation method, which calculates its derivative.
 */
public abstract class Operation extends Node implements Callable<Tensor> {

    /**
     * Constructs an <code>Operation</code> node, adding itself to the consumers of its children.
     *
     * @param children the children of the node
     * @see Node#Node(Node...)
     */
    Operation(Node... children) {
        super(children);

        for (Node child : children) {
            child.getConsumers().add(this);
        }
    }

    /**
     * When this node is called, it will compute and return its output.
     *
     * @return the output of this node, as a tensor
     */
    @Override public Tensor call() {
        return computeOutput();
    }

    /**
     * Computes the output of the node as a tensor, given the inputs of the node.
     *
     * @param inputs the inputs of this node, as tensors
     * @return the output of this node, as a tensor
     */
    protected abstract Tensor computeOutput(Tensor[] inputs);

    /**
     * Calls the method, which computes the output of the node, inputting the outputs of children
     * nodes.
     *
     * @return the output of this node, as a tensor
     */
    @Override protected Tensor computeOutput() {
        return computeOutput(Arrays.stream(children).map(Results::get).toArray(Tensor[]::new));
    }
}
