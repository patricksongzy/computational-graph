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

import graph.exception.NodeComputationException;
import graph.node.leaves.Placeholder;
import math.Tensor;

import java.util.*;
import java.util.concurrent.*;

/**
 * A <code>Graph</code> represents a computational graph. It is used to calculate a graph.node, and its gradient. Graphs are sorted to maximise
 * efficiency.
 */
public class Graph {

    // the executor service is used to execute operations efficiently
    private static final ExecutorService es = Executors.newFixedThreadPool(Runtime.getRuntime().availableProcessors());
    // a list of graphs which have been created
    private static final List<Graph> graphs = new ArrayList<>();
    // the current graph
    private static Graph current = new Graph();

    static {
        // ensure the executor service shuts down
        Runtime.getRuntime().addShutdownHook(new Thread(es::shutdown));
    }

    // a list of nodes in the graph
    private List<Node> nodes = new ArrayList<>();
    private boolean isSorted = false;

    private int outputAmount;
    private Node[] computedNodes;

    /**
     * Constructs a graph and adds it to the list of graphs.
     */
    @SuppressWarnings("WeakerAccess") public Graph() {
        graphs.add(this);
    }

    /**
     * Adds a graph.node to the graph, so it can be tracked an run easily.
     *
     * @param node the graph.node to add
     */
    static void addNode(Node node) {
        current.nodes.add(node);
        current.isSorted = false;
    }

    /**
     * Clears all graphs of nodes and creates a new default graph.
     */
    @SuppressWarnings("WeakerAccess") public static void clearAll() {
        graphs.clear();
        current = new Graph();
    }

    /**
     * Computes the specified nodes, provided the proper placeholders have been inputted. Sorts the
     * graph topologically, then based on the distance of nodes to the output nodes in order to
     * allow for the graph to be computed efficiently.
     *
     * @param placeholderMap the inputted placeholders
     * @param outputNodes    the nodes to compute
     */
    @SuppressWarnings("WeakerAccess") public static void compute(Map<Placeholder, Tensor> placeholderMap, Node... outputNodes) {
        // if no nodes are computed, then return immediately
        if (outputNodes.length == 0) {
            return;
        }

        // add placeholders to the results
        computePlaceholders(placeholderMap);

        if (!current.isSorted) {
            current.nodes = Arrays.asList(current.sortGraph());
        }

        Set<Node> discoverable = new HashSet<>();
        for (Node outputNode : outputNodes) {
            populateDiscoverable(outputNode, discoverable);
        }

        Node[] sorted = current.nodes.stream().filter(discoverable::contains).toArray(Node[]::new);
        current.outputAmount = outputNodes.length;
        current.computedNodes = sorted;

        try {
            for (Node currentNode : sorted) {
                if (currentNode instanceof Operation) {
                    // run the operation using the executor, and submit the result as a future
                    Operation operation = (Operation) currentNode;
                    Results.putOutput(operation, es.submit((Callable<Tensor>) operation::computeOutput));
                } else {
                    // non-operations do not need to be submitted to the executor
                    Results.putOutput(currentNode, CompletableFuture.completedFuture(currentNode.computeOutput()));
                }
            }

            // ensure that all nodes have been computed
            Results.getAllOutputs();
        } catch (InterruptedException | ExecutionException e) {
            es.shutdown();
            throw new NodeComputationException(e);
        }
    }

    /**
     * Starts the computation of the gradient for a graph.node on the graph, given whether it is an end graph.node or not.
     *
     * @param currentNode the graph.node to compute the gradients for
     * @param isEndNode whether the graph.node is an end graph.node or not
     */
    private static void computeGradient(Node currentNode, boolean isEndNode) {
        if (currentNode instanceof Operation) {
            Operation operation = (Operation) currentNode;
            Results.putGradient(operation, es.submit(() -> operation.computeGradients(isEndNode)));
        } else {
            Results.putGradient(currentNode, CompletableFuture.completedFuture(currentNode.computeGradients(isEndNode)));
        }
    }

    /**
     * Computes the placeholder values, by putting the inputted placeholders to the results.
     *
     * @param placeholderMap the inputted placeholders
     */
    private static void computePlaceholders(Map<Placeholder, Tensor> placeholderMap) {
        if (placeholderMap != null) {
            for (Placeholder placeholder : placeholderMap.keySet()) {
                // placeholders do not need to be added from other graphs
                if (current.nodes.contains(placeholder)) {
                    Results.putOutput(placeholder, CompletableFuture.completedFuture(placeholderMap.get(placeholder)));
                }
            }
        }
    }

    /**
     * Returns the currently active graph.
     *
     * @return the currently active graph
     */
    @SuppressWarnings({"unused", "WeakerAccess"}) public static Graph getCurrent() {
        return current;
    }

    /**
     * Gets the default graph.
     *
     * @return the default graph
     */
    @SuppressWarnings("unused") public static Graph getDefault() {
        return graphs.get(0);
    }

    /**
     * Computes the gradient of all nodes which were computed when the graph was last run.
     */
    @SuppressWarnings("WeakerAccess") public static void gradient() {
        // check if the graph was run
        if (current.computedNodes != null) {
            try {
                // compute the gradients of the end nodes
                for (int i = current.computedNodes.length - 1; i >= current.computedNodes.length - current.outputAmount; i--) {
                    Node currentNode = current.computedNodes[i];
                    computeGradient(currentNode, true);
                }

                // compute the gradients of the other nodes
                for (int i = current.computedNodes.length - current.outputAmount - 1; i >= 0; i--) {
                    Node currentNode = current.computedNodes[i];
                    computeGradient(currentNode, false);
                }

                // ensure all gradients have been computed
                Results.getAllGradients();
            } catch (InterruptedException | ExecutionException e) {
                es.shutdown();
                throw new NodeComputationException(e);
            }
        } else {
            throw new IllegalStateException("The graph has not been computed, therefore the gradient cannot be computed.");
        }
    }

    /**
     * Populates a set of discoverable nodes for the topological sort, by adding all descendants of
     * a specific graph.node to the set.
     *
     * @param current      the graph.node to add descendants from
     * @param discoverable the set of discoverable nodes for the topological sort
     */
    private static void populateDiscoverable(Node current, Set<Node> discoverable) {
        discoverable.addAll(Arrays.asList(current.children));
        for (Node child : current.children) {
            populateDiscoverable(child, discoverable);
        }

        discoverable.add(current);
    }

    /**
     * Visits a graph.node, by recursively visiting its descendants, then adding itself to the sorted
     * nodes. This ensures that a graph.node's descendants are placed before the graph.node in the sorted
     * dequeue.
     *
     * @param node         the current graph.node which is being visited
     * @param discoverable the set of discoverable nodes
     * @param sorted       the sorted graph
     */
    private static void visit(Node node, Set<Node> discoverable, Deque<Node> sorted) {
        // check if the graph.node has already been visited and added to the graph
        if (sorted.contains(node)) {
            return;
        }

        // if the graph.node is not discoverable, the graph is not directed
        if (!discoverable.contains(node)) {
            throw new IllegalArgumentException("Unable to sort the graph. Graph is not directed.");
        }

        // remove the graph.node from the discoverable nodes
        discoverable.remove(node);
        // visit the graph.node's ancestors recursively
        for (Node child : node.children) {
            visit(child, discoverable, sorted);
        }

        // add the graph.node to the sorted graph
        sorted.push(node);
    }

    /**
     * Sets this graph to be the current graph.
     *
     * @return this graph
     */
    @SuppressWarnings("WeakerAccess") public Graph setCurrent() {
        current = this;

        return this;
    }

    /**
     * Sorts the graph topologically, then based on distance from end nodes. This ensures that nodes
     * are computed in an efficient order.
     *
     * @param endNodes the end nodes
     * @return the sorted graph
     */
    Node[] sortGraph(Node... endNodes) {
        // the topologically sorted graph
        Deque<Node> topological = new ArrayDeque<>();
        // a set of nodes which may be discovered in the topological sort
        Set<Node> discoverable = new HashSet<>();

        if (endNodes.length == 0) {
            discoverable.addAll(current.nodes);
            isSorted = true;
        } else {
            // populate the discoverable nodes based on only descendants of the end nodes, as the rest do not need to be computed
            for (Node endNode : endNodes) {
                populateDiscoverable(endNode, discoverable);
            }
        }

        // perform the topological sort until no more nodes are discoverable
        while (!discoverable.isEmpty()) {
            visit(discoverable.iterator().next(), discoverable, topological);
        }

        // the distances from the end nodes, where more negative values are considered farther
        Map<Node, Integer> distances = new HashMap<>();
        for (Node node : topological) {
            // because the graph is topologically sorted, the distance can be increased by traversing from end graph.node to start
            distances.put(node, node.getConsumers().stream().mapToInt(consumer -> distances.getOrDefault(consumer, 1) - 1).min().orElse(0));
        }

        // return the nodes, sorted with the farthest nodes from the end first
        return distances.entrySet().stream().sorted(Map.Entry.comparingByValue()).map(Map.Entry::getKey).toArray(Node[]::new);
    }
}
