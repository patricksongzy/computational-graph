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

import neural.graph.node.operation.Operations;
import neural.math.Tensor;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

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
     * Computes the gradient of the node. This method returns a map of gradients, as such gradient computation applies to the non-operation
     * nodes. The map contains the gradients and their respective nodes. When calculating in reverse order, these nodes are the children
     * nodes. In essence, the gradient is calculated with respect to the inputs of the node.
     *
     * @param gradients the map of gradients and their respective nodes
     * @param delta     the gradients with respect to the output of the node
     * @return the map of gradients and respective nodes
     */
    protected Map<Long, Tensor> computeGradients(Map<Long, Tensor> gradients, Tensor delta) {
        return gradients;
    }

    /**
     * Computes the gradient of the node. This method takes the delta as a sum of incoming deltas, or as a tensor of ones, depending on
     * whether it is an output node, then calls the {@link Node#computeGradients(Map, Tensor)} method to compute the gradients, returning
     * the result.
     *
     * @param isEndNode whether the node is an end node or not
     * @return the map of gradients and respective nodes
     */
    Map<Long, Tensor> computeGradients(boolean isEndNode) {
        Tensor delta;

        if (isEndNode)
            delta = Tensor.ones(Results.getOutput(this).getDimensions());
        else
            delta = Operations
                    .addition(getConsumers().stream().map(consumer -> Results.getComputedGradients(consumer, this)).toArray(Tensor[]::new));

        Map<Long, Tensor> gradients = new HashMap<>();
        gradients.put(nodeID, delta);

        return computeGradients(gradients, delta);
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
    List<Operation> getConsumers() {
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

    @Override public String toString() {
        return "id: " + nodeID;
    }
}
