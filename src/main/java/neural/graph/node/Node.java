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

package neural.graph.node;

import java.util.ArrayList;
import java.util.List;
import neural.graph.node.operation.Operation;
import neural.math.Tensor;

/**
 * A <code>Node</code> represents a given operation or value in the computational graph. A
 * <code>Node</code> must have a unique identifier and must keep track of consumers and children.
 */
public abstract class Node {

    // the amount of nodes is used to uniquely identify each node
    private static long nodeCount = 0;
    // the children of the node are used as its inputs
    protected final Node[] children;
    private final long nodeID = nodeCount++;
    // the consumers of the node use its output
    private final List<Operation> consumers = new ArrayList<>();

    /**
     * Constructs a node given its children, adding itself to the graph.
     *
     * @param children the children of the node
     */
    protected Node(Node... children) {
        this.children = children;

        Graph.addNode(this);
    }

    /**
     * Returns the output of the node as a tensor.
     *
     * @return the output of the node
     */
    protected abstract Tensor computeOutput();

    /**
     * Returns the consumers of the node. Consumers use the output of a node as an input.
     *
     * @return the consumers of the node
     */
    public List<Operation> getConsumers() {
        return consumers;
    }

    /**
     * Returns the unique identifier for the node.
     *
     * @return the unique identifier
     */
    public long getID() {
        return nodeID;
    }

    @Override
    public String toString() {
        return "id: " + nodeID;
    }
}
