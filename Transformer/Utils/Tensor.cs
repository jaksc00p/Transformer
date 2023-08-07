using System;

namespace Transformer.Utils
{
    public class Tensor
    {
        private int nrValues;
        private int[] sizes;
        private Rev[] values;

        public int Dimension { get { return sizes.Length; } }

        public static Tensor operator *(Tensor T1, Tensor T2)
        {
            return MatMulElementWise(T1, T2);
        }

        public static Tensor operator *(Tensor T, double d)
        {
            return T.Scale(d);
        }

        public static Tensor operator *(double d, Tensor T)
        {
            return T * d;
        }

        public static Tensor operator /(Tensor T1, Tensor T2)
        {
            return MatDivElementWise(T1, T2);
        }

        public static Tensor operator /(Tensor T, double d)
        {
            return T.Scale(1.0 / d);
        }

        public static Tensor operator +(Tensor T1, Tensor T2)
        {
            return MatAdd(T1, T2);
        }

        public static Tensor operator +(Tensor T, double d)
        {
            return T.Add(d);
        }

        public static Tensor operator +(double d, Tensor T)
        {
            return T + d;
        }

        public static Tensor operator -(Tensor T1, Tensor T2)
        {
            return MatAdd(T1, T2 * -1);
        }

        public static Tensor operator -(Tensor T, double d)
        {
            return T.Add(-d);
        }

        public static Tensor operator -(double d, Tensor T)
        {
            return d + -1 * T;
        }

        /// <summary>
        /// Index parameter implementation
        /// </summary>
        /// <param name="keys"></param>
        /// <returns></returns>
        /// <exception cref="ArgumentException"></exception>
        public Rev this[params int[] keys]
        {
            get
            {
                if (keys.Length != Dimension)
                    throw new ArgumentException("Wrong number of dimensions");
                for (int dim = 0; dim < Dimension; dim++)
                {
                    if (keys[dim] < 0)
                        throw new ArgumentException("key must be nonnegative");
                    if (keys[dim] >= sizes[dim])
                        throw new ArgumentException("key is outside the dimension of the tensor");
                }

                int ind = 0;
                int blocksize = 1;
                for (int dim = Dimension - 1; dim >= 0; dim--)
                {
                    ind += keys[dim] * blocksize;
                    blocksize *= sizes[dim];
                }

                return values[ind];
            }
            set
            {
                if (keys.Length != Dimension)
                    throw new ArgumentException("Wrong number of dimensions");
                for (int dim = 0; dim < Dimension; dim++)
                {
                    if (keys[dim] < 0)
                        throw new ArgumentException("key must be nonnegative");
                    if (keys[dim] >= sizes[dim])
                        throw new ArgumentException("key is outside the dimension of the tensor");
                }

                int ind = 0;
                int blocksize = 1;
                for (int dim = Dimension - 1; dim >= 0; dim--)
                {
                    ind += keys[dim] * blocksize;
                    blocksize *= sizes[dim];
                }

                values[ind] = value;
            }
        }

        /// <summary>
        /// Constructor
        /// </summary>
        /// <param name="sizes"></param>
        /// <exception cref="ArgumentNullException"></exception>
        /// <exception cref="ArgumentException"></exception>
        public Tensor(params int[] sizes)
        {
            if (sizes == null)
                throw new ArgumentNullException(nameof(sizes));
            if (sizes.Length == 0)
                throw new ArgumentException(nameof(sizes) + "Must have at least one dimension");

            this.sizes = new int[sizes.Length];
            sizes.CopyTo(this.sizes, 0);

            nrValues = 1;
            for (int dim = 0; dim < Dimension; dim++)
            {
                if (sizes[dim] < 1)
                    throw new ArgumentException(nameof(sizes) + "Size of dimension must be > 0");
                nrValues *= sizes[dim];
            }

            values = new Rev[nrValues];
            for (int c = 0; c < nrValues; c++)
            {
                values[c] = new Rev(0.0);
            }
        }

        /// <summary>
        /// Copy constructor. Removes the derivatives and actions.
        /// </summary>
        /// <param name="T"></param>
        public Tensor(Tensor T, bool copy = false)
        {
            sizes = new int[T.Dimension];
            Array.Copy(T.sizes, sizes, Dimension);

            nrValues = 1;
            for (int dim = 0; dim < Dimension; dim++)
            {
                nrValues *= sizes[dim];
            }

            values = new Rev[nrValues];
            for (int c = 0; c < nrValues; c++)
            {
                if (copy)
                    values[c] = T.values[c];
                else
                    values[c] = new Rev(T.values[c].Magnitude);
            }
        }

        /// <summary>
        /// Transpose the last two dimensions
        /// </summary>
        /// <returns></returns>
        public Tensor Transpose()
        {
            if (Dimension < 2)
                throw new ArgumentException("Tensor must have dimension >= 2");

            Tensor T = new Tensor(sizes);

            T.sizes[Dimension - 2] = sizes[Dimension - 1];
            T.sizes[Dimension - 1] = sizes[Dimension - 2];

            int imax = T.sizes[Dimension - 2];
            int jmax = T.sizes[Dimension - 1];

            int nrBlocks = nrValues / (imax * jmax);
            for (int block = 0; block < nrBlocks; block++)
            {
                int offset = block * imax * jmax;
                for (int i = 0; i < imax; i++)
                {
                    for (int j = 0; j < jmax; j++)
                    {
                        T.values[offset + jmax * i + j] = values[offset + imax * j + i];
                    }
                }
            }

            return T;
        }

        /// <summary>
        /// Scale all elements with a number
        /// </summary>
        /// <param name="s"></param>
        public Tensor Scale(double s)
        {
            Tensor T = new Tensor(this, true);

            for (int c = 0; c < nrValues; c++)
            {
                T.values[c] = values[c] * s;
            }

            return T;
        }

        /// <summary>
        /// Add a number to all elements
        /// </summary>
        /// <param name="s"></param>
        public Tensor Add(double s)
        {
            Tensor T = new Tensor(this, true);

            for (int c = 0; c < nrValues; c++)
            {
                T.values[c] = values[c] + s;
            }

            return T;
        }

        /// <summary>
        /// Element-wise power 
        /// </summary>
        /// <param name="e"></param>
        /// <returns></returns>
        public Tensor Pow(double e)
        {
            Tensor T = new Tensor(this, true);

            for (int c = 0; c < nrValues; c++)
            {
                T.values[c] = values[c].Pow(e);
            }

            return T;
        }

        /// <summary>
        /// Generate normal random values with He scaling over the second last dimension of the tensor.
        /// </summary>
        public void GenerateNormalRandomValues()
        {
            if (Dimension < 2)
                throw new ArgumentException("Tensor must have dimension >= 2");

            int imax = sizes[Dimension - 2];
            for (int c = 0; c < nrValues; c++)
            {
                values[c] = new Rev((double)(RandomNumbers.Instance.GetNextNormalNumber() * Math.Sqrt(2.0 / imax)));
            }
        }

        /// <summary>
        /// Softmax over last dimension
        /// </summary>
        public Tensor Softmax()
        {
            Tensor T = new Tensor(this, true);

            int imax = sizes[Dimension - 1];
            int nrBlocks = nrValues / imax;
            for (int block = 0; block < nrBlocks; block++)
            {
                int offset = block * imax;
                Rev normalization = new Rev(0);
                for (int i = 0; i < imax; i++)
                {
                    normalization += values[offset + i].Exp();
                }
                for (int i = 0; i < imax; i++)
                {
                    T.values[offset + i] = values[offset + i].Exp() / normalization;
                }
            }

            return T;
        }

        /// <summary>
        /// Mask upper triangular part of the last two dimensions of a tensor
        /// </summary>
        /// <exception cref="ArgumentException"></exception>
        public void Mask()
        {
            if (Dimension < 2)
                throw new ArgumentException("Tensor must have dimension >= 2");

            int imax = sizes[Dimension - 2];
            int jmax = sizes[Dimension - 1];

            int nrBlocks = nrValues / (imax * jmax);
            for (int block = 0; block < nrBlocks; block++)
            {
                int offset = block * imax * jmax;
                for (int i = 0; i < imax; i++)
                {
                    for (int j = i + 1; j < jmax; j++)
                    {
                        values[offset + jmax * i + j] = new Rev(double.NegativeInfinity);
                    }
                }
            }

        }

        /// <summary>
        /// Zero out elemens of the last dimension of a tensor based on a masking array
        /// </summary>
        /// <param name="dropoutMask"></param>
        /// <param name="isTraining"></param>
        /// <exception cref="ArgumentException"></exception>
        public Tensor Dropout(bool[] dropoutMask, double dropoutRate)
        {
            int imax = sizes[Dimension - 1];
            if (imax != dropoutMask.Length)
                throw new ArgumentException("Wrong length of fropout vector");

            Tensor T = new Tensor(this, true);
            T *= 1.0 / (1.0 - dropoutRate);

            int nrBlocks = nrValues / imax;
            for (int block = 0; block < nrBlocks; block++)
            {
                int offset = block * imax;
                for (int i = 0; i < imax; i++)
                {
                    if (dropoutMask[i])
                        T.values[offset + i] = new Rev(0.0);
                }
            }

            return T;

        }

        /// <summary>
        /// Set all elements < 0 to 0
        /// </summary>
        public void ReLU()
        {
            for (int c = 0; c < nrValues; c++)
            {
                if (values[c].Magnitude < 0)
                    values[c] = new Rev(0.0);
            }
        }

        /// <summary>
        /// Flatten last two dimensions. 
        /// </summary>
        /// <param name="reduceDimension"></param>
        /// <returns></returns>
        /// <exception cref="ArgumentException"></exception>
        public Tensor Flatten()
        {
            if (Dimension < 2)
                throw new ArgumentException("Tensor must have dimension >= 2");

            int[] sizes = new int[Dimension];
            if (Dimension > 2)
                Array.Copy(this.sizes, sizes, this.sizes.Length - 2);
            int imax = this.sizes[Dimension - 2];
            int jmax = this.sizes[Dimension - 1];
            sizes[Dimension - 2] = 1;
            sizes[Dimension - 1] = imax * jmax;
            Tensor C = new Tensor(sizes);

            int nrBlocks = nrValues / (imax * jmax);
            for (int block = 0; block < nrBlocks; block++)
            {
                int offset = block * imax * jmax;
                int ind = 0;
                for (int i = 0; i < imax; i++)
                {
                    for (int j = 0; j < jmax; j++)
                    {
                        C.values[offset + ind++] = values[offset + i * jmax + j];
                    }
                }
            }

            return C;
        }

        /// <summary>
        /// Only used for final output layer
        /// </summary>
        /// <returns></returns>
        /// <exception cref="InvalidOperationException"></exception>
        public int[] GetMaxIndex()
        {
            if (Dimension != 3)
                throw new InvalidOperationException("Tensor must have dimension = 3");
            if (sizes[1] != 1)
                throw new InvalidOperationException("Second dimension must have size = 1");

            int imax = sizes[0];
            int jmax = sizes[2];
            int[] maxind = new int[imax];

            for (int i = 0; i < imax; i++)
            {
                double maxval = 0;
                for (int j = 0; j < jmax; j++)
                {
                    if (values[i * jmax + j].Magnitude > maxval)
                    {
                        maxind[i] = j;
                        maxval = values[i * jmax + j].Magnitude;
                    }
                }
            }

            return maxind;
        }



        /// <summary>
        /// Add a vector to the last dimension of the matrix
        /// </summary>
        /// <param name="v"></param>
        public Tensor VecAdd(Tensor v)
        {
            Tensor T = new Tensor(this, true);

            if (v.Dimension != 1)
                throw new ArgumentException("Vector must have dimension 1");
            if (v.sizes[0] != sizes[Dimension - 1])
                throw new ArgumentException("Wrong size of vector");

            int imax = sizes[Dimension - 1];
            int nrBlocks = nrValues / imax;
            for (int block = 0; block < nrBlocks; block++)
            {
                int offset = block * imax;
                for (int i = 0; i < imax; i++)
                {
                    T.values[offset + i] += v.values[i];
                }
            }

            return T;

        }

        /// <summary>
        /// In place element-wise addition of the elemens of two tensors with identical shape.
        /// Only used for optimizer.
        /// </summary>
        /// <param name="T"></param>
        /// <exception cref="ArgumentException"></exception>
        public void MatAdd(Tensor T)
        {
            if (Dimension != T.Dimension)
                throw new ArgumentException("Tensors must have >= 2 dimensions");
            for (int dim = 0; dim < Dimension; dim++)
            {
                if (sizes[dim] != T.sizes[dim])
                    throw new ArgumentException("Dimensions of tensors must be identical");
            }

            for (int c = 0; c < nrValues; c++)
            {
                values[c] += T.values[c];
            }

        }

        /// <summary>
        /// Remove derivatives and derivative actions
        /// </summary>
        public void ClearDerivatives()
        {
            for (int c = 0; c < nrValues; c++)
            {
                values[c] = new Rev(values[c].Magnitude);
            }
        }

        /// <summary>
        /// Extract derivatives of a tensor and return as a new tensor
        /// </summary>
        /// <returns></returns>
        public Tensor GetDerivatives()
        {
            Tensor T = new Tensor(sizes);

            for (int c = 0; c < nrValues; c++)
            {
                T.values[c] = new Rev(values[c].Derivative);
            }

            return T;
        }

        /// <summary>
        /// Perform derivative calculations for all elements in a tensor with input values from 
        /// the derivatives of another tensor with the same shape
        /// </summary>
        /// <param name="T"></param>
        /// <exception cref="ArgumentException"></exception>
        public void TransferDerivatives(Tensor T)
        {
            if (Dimension != T.Dimension)
                throw new ArgumentException("Tensors must have the same number of dimensions");
            for (int dim = 0; dim < Dimension; dim++)
            {
                if (sizes[dim] != T.sizes[dim])
                    throw new ArgumentException("Size of dimensions must be equal");
            }

            for (int c = 0; c < nrValues; c++)
            {
                if (T.values[c].Derivative != 0)
                    values[c].CalculateDerivative(T.values[c].Derivative);
            }

        }

        /// <summary>
        /// Element-wise addition of the elemens of two tensors with identical shape
        /// </summary>
        /// <param name="A"></param>
        /// <param name="B"></param>
        /// <returns></returns>
        /// <exception cref="ArgumentException"></exception>
        public static Tensor MatAdd(Tensor A, Tensor B)
        {
            if (A.Dimension != B.Dimension)
                throw new ArgumentException("Tensors must have >= 2 dimensions");
            for (int dim = 0; dim < A.Dimension; dim++)
            {
                if (A.sizes[dim] != B.sizes[dim])
                    throw new ArgumentException("Dimensions of tensors must be identical");
            }

            Tensor C = new Tensor(A.sizes);
            for (int c = 0; c < C.nrValues; c++)
            {
                C.values[c] = A.values[c] + B.values[c];
            }

            return C;

        }

        /// <summary>
        /// Element-wise division of the elemens of two tensors with identical shape
        /// </summary>
        /// <param name="A"></param>
        /// <param name="B"></param>
        /// <returns></returns>
        public static Tensor MatDivElementWise(Tensor A, Tensor B)
        {
            if (A.Dimension != B.Dimension)
                throw new ArgumentException("Tensors must have >= 2 dimensions");
            for (int dim = 0; dim < A.Dimension; dim++)
            {
                if (A.sizes[dim] != B.sizes[dim])
                    throw new ArgumentException("Dimensions of tensors must be identical");
            }

            Tensor C = new Tensor(A.sizes);
            for (int c = 0; c < C.nrValues; c++)
            {
                C.values[c] = A.values[c] / B.values[c];
            }

            return C;

        }

        /// <summary>
        /// Element-wise multiplication of the elemens of two tensors with identical shape
        /// </summary>
        /// <param name="A"></param>
        /// <param name="B"></param>
        /// <returns></returns>
        public static Tensor MatMulElementWise(Tensor A, Tensor B)
        {
            if (A.Dimension != B.Dimension)
                throw new ArgumentException("Tensors must have >= 2 dimensions");
            for (int dim = 0; dim < A.Dimension; dim++)
            {
                if (A.sizes[dim] != B.sizes[dim])
                    throw new ArgumentException("Dimensions of tensors must be identical");
            }

            Tensor C = new Tensor(A.sizes);
            for (int c = 0; c < C.nrValues; c++)
            {
                C.values[c] = A.values[c] * B.values[c];
            }

            return C;

        }

        /// <summary>
        /// Matrix multiplication over the last two dimensions of two tensors
        /// </summary>
        /// <param name="A"></param>
        /// <param name="B"></param>
        /// <returns></returns>
        public static Tensor MatMul(Tensor A, Tensor B)
        {
            if (A.Dimension < 2 || B.Dimension < 2)
                throw new ArgumentException("Tensors must have >= 2 dimensions");
            if (A.sizes[A.Dimension - 1] != B.sizes[B.Dimension - 2])
                throw new ArgumentException("Wrong dimensions for matrix multiplication");

            int imax = A.sizes[A.Dimension - 2];
            int jmax = B.sizes[B.Dimension - 1];
            int kmax = A.sizes[A.Dimension - 1];
            int blockSizeA = imax * kmax;
            int blockSizeB = jmax * kmax;
            int blockSizeC = jmax * imax;

            if (B.Dimension > 2 && A.nrValues / blockSizeA != B.nrValues / blockSizeB)
                throw new ArgumentException("Wrong dimensions for matrix multiplication");

            int[] sizes = new int[A.Dimension];
            A.sizes.CopyTo(sizes, 0);
            sizes[A.Dimension - 2] = imax;
            sizes[A.Dimension - 1] = jmax;
            Tensor C = new Tensor(sizes);

            if (B.Dimension == 2)
                blockSizeB = 0;

            int nrblocks = C.nrValues / (imax * jmax);
            for (int block = 0; block < nrblocks; block++)
            {
                int offsetA = block * blockSizeA;
                int offsetB = block * blockSizeB;
                int offsetC = block * blockSizeC;

                for (int i = 0; i < imax; i++)
                {
                    for (int j = 0; j < jmax; j++)
                    {
                        for (int k = 0; k < kmax; k++)
                        {
                            C.values[offsetC + i * jmax + j] += A.values[offsetA + i * kmax + k] * B.values[offsetB + k * jmax + j];
                        }
                    }
                }
            }

            C = Checkpoints.Instance.AddCheckpoint(C);

            return C;

        }

        /// <summary>
        /// Concatenate an array of tensors of identical shape along its last dimension
        /// </summary>
        /// <param name="Tensors"></param>
        /// <returns></returns>
        public static Tensor Concat(Tensor[] Tensors)
        {
            for (int t = 1; t < Tensors.Length; t++)
            {
                if (Tensors[t].Dimension != Tensors[0].Dimension)
                    throw new ArgumentException("All tensors in array must have the same dimension");
                for (int dim = 0; dim < Tensors[0].Dimension; dim++)
                {
                    if (Tensors[t].sizes[dim] != Tensors[0].sizes[dim])
                        throw new ArgumentException("All tensors dimensions in array must be equal");
                }
            }

            int[] sizes = new int[Tensors[0].Dimension];
            Tensors[0].sizes.CopyTo(sizes, 0);
            int lastdim = sizes[sizes.Length - 1];
            sizes[sizes.Length - 1] *= Tensors.Length;
            Tensor C = new Tensor(sizes);

            int nrBlocks = Tensors[0].nrValues / lastdim;
            for (int block = 0; block < nrBlocks; block++)
            {
                int offsetT = block * lastdim;
                int offsetC = block * lastdim * Tensors.Length;
                for (int t = 1; t < Tensors.Length; t++)
                {
                    for (int i = 0; i < lastdim; i++)
                    {
                        C.values[offsetC + t * lastdim + i] = Tensors[t].values[offsetT + i];
                    }
                }
            }

            return C;

        }

        /// <summary>
        /// Add two tensors and normalize over last dimension
        /// </summary>
        /// <param name="A"></param>
        /// <param name="B"></param>
        /// <returns></returns>
        public static Tensor AddNorm(Tensor A, Tensor B)
        {
            if (A.Dimension != B.Dimension)
                throw new ArgumentException("Tensors must have the same number of dimensions");

            double eps = 0.001;

            Tensor C = new Tensor(A);

            int imax = C.sizes[C.Dimension - 1];
            int nrBlocks = C.nrValues / imax;
            for (int block = 0; block < nrBlocks; block++)
            {
                int offset = block * imax;
                Rev mean = new Rev(0);
                for (int i = 0; i < imax; i++)
                {
                    mean += (A.values[offset + i] + B.values[offset + i]) / imax;
                }

                Rev var = new Rev(0);
                for (int i = 0; i < imax; i++)
                {
                    var += (A.values[offset + i] + B.values[offset + i] - mean).Pow(2) / imax;
                }

                for (int i = 0; i < imax; i++)
                {
                    C.values[offset + i] = (A.values[offset + i] + B.values[offset + i] - mean) / (var + eps).Pow(0.5);
                }
            }

            C = Checkpoints.Instance.AddCheckpoint(C);

            return C;
        }

    }
}
