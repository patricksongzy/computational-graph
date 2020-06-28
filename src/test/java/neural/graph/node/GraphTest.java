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

import neural.graph.node.leaves.Constant;
import neural.graph.node.leaves.Placeholder;
import neural.graph.node.operation.Addition;
import neural.graph.node.operation.Multiplication;
import neural.math.Tensor;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.Test;

import java.util.HashMap;
import java.util.Map;

import static org.assertj.core.api.Assertions.assertThat;

class GraphTest {
    /**
     * Tests the gradient calculation from addition of tensors with compatible, but non-matching dimensions.
     */
    @Test void addBroadcastGradientsTest() {
        Constant a = new Constant(new Tensor.Builder(2, 3).setValues(3, 8, 2, 5, 1, 6).build());
        Constant b = new Constant(new Tensor.Builder(1, 3).setValues(3, 2, 1).build());

        Addition c = new Addition(a, b);

        Graph.compute(null, c);
        Graph.gradient();

        assertThat(Results.getGradient(a)).isEqualTo(new Tensor.Builder(2, 3).setValues(1, 1, 1, 1, 1, 1).build());
        assertThat(Results.getGradient(b)).isEqualTo(new Tensor.Builder(1, 3).setValues(2, 2, 2).build());
    }

    /**
     * Tests the gradient calculation of a specific node.
     */
    @Test void executeNodeTest() {
        Placeholder a = new Placeholder();
        Placeholder b = new Placeholder();
        Placeholder c = new Placeholder();

        Map<Placeholder, Tensor> placeholderMap = new HashMap<>();
        placeholderMap.put(a, new Tensor.Builder(1).setValues(2).build());
        placeholderMap.put(b, new Tensor.Builder(3).setValues(1, 5, 6).build());
        placeholderMap.put(c, new Tensor.Builder(3).setValues(3, 2, 8).build());

        Addition d = new Addition(a, a);
        Multiplication e = new Multiplication(a, b, d);

        Graph.compute(placeholderMap, e);

        Tensor expected = new Tensor.Builder(3).setValues(8, 40, 48).build();
        assertThat(Results.getOutput(e)).isEqualTo(expected);
    }

    /**
     * Tests the gradient calculation of several chained operations.
     */
    @Test void gradientsTest() {
        Constant a = new Constant(2);
        Constant b = new Constant(1);
        Constant one = new Constant(1);

        Addition c = new Addition(a, b);
        Addition d = new Addition(b, one);

        Multiplication e = new Multiplication(c, d);

        Graph.compute(null, e);

        Graph.gradient();

        assertThat(Results.getGradient(a).getValues()).isEqualTo(new float[] {2});
        assertThat(Results.getGradient(b).getValues()).isEqualTo(new float[] {5});
    }

    /**
     * Tests that the presence of another graph does not affect the computation of some graph.
     */
    @Test void multipleGraphsTest() {
        Placeholder a = new Placeholder();
        Placeholder b = new Placeholder();
        Placeholder c = new Placeholder();

        Map<Placeholder, Tensor> placeholderMap = new HashMap<>();
        placeholderMap.put(a, new Tensor.Builder(1).setValues(2).build());
        placeholderMap.put(b, new Tensor.Builder(3).setValues(1, 5, 6).build());
        placeholderMap.put(c, new Tensor.Builder(3).setValues(3, 2, 8).build());

        Addition d = new Addition(a, a);
        Multiplication e = new Multiplication(a, b, d);
        Addition f = new Addition(a, c);

        // a * b * (a + a) + a + c
        Graph.compute(placeholderMap, e, f);

        Graph graph = new Graph().setCurrent();

        Placeholder pd = new Placeholder();
        Placeholder pe = new Placeholder();
        Placeholder pf = new Placeholder();

        Multiplication g = new Multiplication(pd, pf);
        Addition h = new Addition(pe, pf);

        Multiplication i = new Multiplication(g, h);

        graph.setCurrent();

        placeholderMap.clear();
        placeholderMap.put(pd, Results.getOutput(d));
        placeholderMap.put(pe, Results.getOutput(e));
        placeholderMap.put(pf, Results.getOutput(f));

        // pd * pf * (pe + pf)
        Graph.compute(placeholderMap, i);

        Tensor expected = new Tensor.Builder(3).setValues(260, 704, 2320).build();
        assertThat(Results.getOutput(i)).isEqualTo(expected);
    }

    /**
     * Tests the gradient calculation from multiplication of tensors with compatible, but non-matching dimensions.
     */
    @Test void multiplyBroadcastGradientsTest() {
        Constant a = new Constant(new Tensor.Builder(2, 3).setValues(3, 8, 2, 5, 1, 6).build());
        Constant b = new Constant(new Tensor.Builder(1, 3).setValues(3, 2, 1).build());

        Multiplication c = new Multiplication(a, b);

        Graph.compute(null, c);
        Graph.gradient();

        assertThat(Results.getGradient(a)).isEqualTo(new Tensor.Builder(2, 3).setValues(3, 2, 1, 3, 2, 1).build());
        assertThat(Results.getGradient(b)).isEqualTo(new Tensor.Builder(1, 3).setValues(8, 9, 8).build());
    }

    /**
     * Tests the calculation of separate trees to ensure that the nodes calculate properly and to ensure the graphs sort properly.
     */
    @Test void separateTreeTest() {
        Constant a = new Constant(new Tensor.Builder(3).setValues(3, 2, 1).build());
        Constant b = new Constant(new Tensor.Builder(3).setValues(1, 2, 1).build());
        Constant c = new Constant(new Tensor.Builder(3).setValues(1, 3, 2).build());
        Constant d = new Constant(new Tensor.Builder(3).setValues(1, 2, 3).build());

        Addition e = new Addition(a, b);
        Addition f = new Addition(c, d);

        Graph.compute(new HashMap<>(), e, f);
        assertThat(Results.getOutput(e)).isEqualTo(new Tensor.Builder(3).setValues(4, 4, 2).build());
        assertThat(Results.getOutput(f)).isEqualTo(new Tensor.Builder(3).setValues(2, 5, 5).build());
    }

    /**
     * Tears down after each test. All graphs are cleared, in order to ensure that nodes which are not used do not affect results.
     */
    @AfterEach void teardown() {
        Graph.clearAll();
    }

    /**
     * Test to ensure that an unused node is properly sorted out.
     */
    @Test void unusedNodeTest() {
        Placeholder a = new Placeholder();
        Placeholder b = new Placeholder();
        Placeholder c = new Placeholder();

        Addition d = new Addition(a, a);
        Multiplication e = new Multiplication(a, b, d);
        Addition f = new Addition(a, c);

        Multiplication g = new Multiplication(d, f);
        Addition h = new Addition(e, f);

        new Multiplication(g, h);

        Node[] sorted = Graph.getCurrent().sortGraph(e, g);

        long minID = a.getID();
        for (int i = 0; i < 2; i++)
            assertThat(sorted[i].getID() - minID).isIn(0L, 2L);

        for (int i = 2; i < 5; i++)
            assertThat(sorted[i].getID() - minID).isIn(1L, 3L, 5L);
    }
}
