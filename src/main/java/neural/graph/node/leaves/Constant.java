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

package neural.graph.node.leaves;

import neural.graph.node.Node;
import neural.math.Tensor;

/**
 * A <code>Constant</code> represents a node whose value will not change during runtime.
 */
public class Constant extends Node {

    // the values of the constant
    private final Tensor values;

    /**
     * Constructs a <code>Constant</code> node from a tensor.
     *
     * @param values the tensor values of the constant
     * @see Node#Node(Node...)
     */
    public Constant(Tensor values) {
        super();

        this.values = values;
    }

    /**
     * Constructs a <code>Constant</code> node, by creating a single-value tensor.
     *
     * @param value the value of the constant
     * @see Node#Node(Node...)
     */
    public Constant(float value) {
        super();

        this.values = new Tensor.Builder(1).setValues(value).build();
    }

    /**
     * Returns the constant as a tensor.
     *
     * @return the constant, as a tensor
     */
    @Override protected Tensor computeOutput() {
        return values;
    }
}
