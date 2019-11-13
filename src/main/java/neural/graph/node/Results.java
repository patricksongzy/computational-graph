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

import neural.graph.exception.NodeComputationException;
import neural.math.Tensor;

import java.util.HashMap;
import java.util.Map;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.Future;

/**
 * The <code>Results</code> represents a map of the results of all computational graphs, stored by
 * node ID.
 */
public class Results {

    // the results are stored based on node ID
    private static final Map<Long, Future<Tensor>> results = new HashMap<>();

    /**
     * Clears any stored results.
     */
    @SuppressWarnings("unused") public static void clear() {
        results.clear();
    }

    /**
     * Gets the output of a node.
     *
     * @param node the node whose output to get
     * @return the output of the node
     * @throws IllegalArgumentException if attempting to retrieve a value which does not exist or
     *                                  has not yet been computed
     */
    public static Tensor get(Node node) {
        if (!results.containsKey(node.getID())) {
            throw new IllegalArgumentException("Attempting to retrieve value which has not yet been computed.");
        }

        try {
            return results.get(node.getID()).get();
        } catch (InterruptedException | ExecutionException e) {
            throw new NodeComputationException(e);
        }
    }

    /**
     * Ensures all values stored in the results have been calculated.
     *
     * @throws ExecutionException   if any exceptions are thrown during execution
     * @throws InterruptedException if any executions are interrupted
     */
    static void getAll() throws ExecutionException, InterruptedException {
        for (Future<Tensor> future : results.values()) {
            future.get();
        }
    }

    /**
     * Adds the output of a node, alongside with the respective ID to the results.
     *
     * @param node  the node whose output to store
     * @param value the output of the node
     */
    static void put(Node node, Future<Tensor> value) {
        results.put(node.getID(), value);
    }
}
