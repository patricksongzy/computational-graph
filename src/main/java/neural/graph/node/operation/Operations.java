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

import neural.math.Tensor;

/**
 * <code>Operations</code> applies an operation to given tensor inputs, without adding the tensors to the computational graph.
 */
public class Operations {
    /**
     * Element-wise adds the given tensors.
     *
     * @param inputs the tensors to add
     * @return the sum of the tensors
     */
    public static Tensor addition(Tensor... inputs) {
        inputs = broadcastInputs(inputs);

        // create an output tensor with the output dimensions
        Tensor result = Tensor.zeros(inputs[0].getDimensions());
        for (int i = 0; i < result.getLength(); i++) {
            float sum = 0;

            // add each tensor element-wise
            // because they are broadcast, this may be done by absolute index
            for (Tensor input : inputs) {
                sum += input.get(i);
            }

            // set the proper value in the result
            result.set(sum, i);
        }

        return result;
    }

    /**
     * Checks if the inputted tensors require broadcasting, then broadcasts if necessary.
     *
     * @param inputs the tensors to broadcast
     * @return the broadcasted tensors
     */
    private static Tensor[] broadcastInputs(Tensor... inputs) {
        if (inputs.length == 0)
            throw new IllegalArgumentException("Cannot compute operation: no inputs provided.");

        // check if the tensors require broadcasting, then broadcast if necessary
        if (Tensor.isDimensionsMismatch(inputs))
            inputs = Tensor.broadcast(inputs);

        return inputs;
    }

    /**
     * Creates a broadcasted input array given a tensor to which an operation is applied to given an array of other tensors.
     *
     * @param start the tensor which the operation is applied to
     * @param end the tensors to apply the operation using
     * @return the broadcasted input array
     */
    private static Tensor[] createInputArray(Tensor start, Tensor... end) {
        Tensor[] inputs = new Tensor[end.length + 1];
        inputs[0] = start;
        System.arraycopy(end, 0, inputs, 1, end.length);

        inputs = broadcastInputs(inputs);
        return inputs;
    }

    /**
     * Element-wise divides a given tensor by an array of tensors.
     *
     * @param numerator the tensor which will be divided
     * @param denominators the tensors which will divide the given tensor
     * @return the divided tensor
     */
    @SuppressWarnings("WeakerAccess") public static Tensor division(Tensor numerator, Tensor... denominators) {
        if (denominators.length == 0)
            return numerator;

        Tensor[] inputs = createInputArray(numerator, denominators);

        // create an output tensor with the output dimensions
        Tensor result = Tensor.zeros(inputs[0].getDimensions());
        for (int i = 0; i < result.getLength(); i++) {
            float quotient = inputs[0].get(i);

            // divide each tensor element-wise
            // because they are broadcast, this may be done by absolute index
            for (int j = 1; j < inputs.length; j++) {
                quotient /= inputs[j].get(i);
            }

            // set the proper value in the result
            result.set(quotient, i);
        }

        return result;
    }

    /**
     * Element-wise multiplies the given tensors.
     *
     * @param inputs the tensors to multiply
     * @return the multiplied tensor
     */
    @SuppressWarnings("WeakerAccess") public static Tensor multiplication(Tensor... inputs) {
        inputs = broadcastInputs(inputs);

        // create an output tensor with the output dimensions
        Tensor result = Tensor.zeros(inputs[0].getDimensions());
        for (int i = 0; i < result.getLength(); i++) {
            float product = 1;

            // multiply each tensor element-wise
            // because they are broadcast, this may be done by absolute index
            for (Tensor input : inputs) {
                product *= input.get(i);
            }

            // set the proper value in the result
            result.set(product, i);
        }

        return result;
    }

    /**
     * Element-wise subtracts an array of tensors from a given tensor.
     *
     * @param minuend the tensor to subtract from
     * @param subtrahend the tensors to subtract
     * @return the subtracted tensor
     */
    @SuppressWarnings("WeakerAccess") public static Tensor subtraction(Tensor minuend, Tensor... subtrahend) {
        if (subtrahend.length == 0)
            return minuend;

        Tensor[] inputs = createInputArray(minuend, subtrahend);

        // create an output tensor with the output dimensions
        Tensor result = Tensor.zeros(inputs[0].getDimensions());
        for (int i = 0; i < result.getLength(); i++) {
            float difference = inputs[0].get(i);

            // subtract each tensor element-wise
            // because they are broadcast, this may be done by absolute index
            for (int j = 1; j < inputs.length; j++) {
                difference -= inputs[j].get(i);
            }

            // set the proper value in the result
            result.set(difference, i);
        }

        return result;
    }

    /**
     * Computes a sum of a tensor along a specific axis.
     *
     * @param input the tensor to sum
     * @param axes the axis to sum along
     * @return the tensor summed along the axis
     */
    public static Tensor sum(Tensor input, int... axes) {
        // the reduced dimensions from taking the sum along an axis
        int[] newDimensions = new int[input.getDimensions().length];
        System.arraycopy(input.getDimensions(), 0, newDimensions, 0, newDimensions.length);

        for (int axis : axes) {
            if (axis < 0 || axis >= newDimensions.length)
                throw new IllegalArgumentException("Cannot sum along axis which does not exist in original tensor.");

            // if a sum is taken along the axis, the dimension length will be one
            newDimensions[axis] = 1;
        }

        Tensor result = Tensor.zeros(newDimensions);
        for (int i = 0; i < input.getLength(); i++) {
            // get the indices for the individual dimensions
            int[] indices = Tensor.getExpandedIndices(input.getDimensions(), i);

            // if the dimension along the axis is summed, then just set the index to zero and keep adding to it in the result
            for (int axis : axes)
                indices[axis] = 0;

            // increment the result given the partially zeroed indices
            result.increment(input.get(i), indices);
        }

        return result;
    }
}
