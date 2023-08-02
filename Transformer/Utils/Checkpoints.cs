using System;

namespace Transformer.Utils
{
    /// <summary>
    /// Increases performence by saving states that the backtracking 
    /// derivative calculations can be performed between.
    /// </summary>
    public sealed class Checkpoints
    {
        private static readonly Checkpoints instance = new Checkpoints();
        public static Checkpoints Instance
        {
            get
            {
                return instance;
            }
        }

        // Explicit static constructor to tell C# compiler not to mark type as beforefieldinit
        static Checkpoints() { }

        private Checkpoints() { }

        private int nr = 1;
        private SortedDictionary<int, Tensor> checkpoints_in = new SortedDictionary<int, Tensor>();
        private SortedDictionary<int, Tensor> checkpoints_out = new SortedDictionary<int, Tensor>();

        public Tensor AddCheckpoint(Tensor data)
        {
            checkpoints_in[nr] = data;
            var data_copy = new Tensor(data);
            checkpoints_out[nr] = data_copy;
            nr++;

            return data_copy;
        }

        public void ClearCheckpoints()
        {
            checkpoints_in.Clear();
            checkpoints_out.Clear();
            nr = 1;
        }

        /// <summary>
        /// Perform derivative calculations for all elements in a tensor with input values from 
        /// the derivatives of another tensor with the same shape
        /// </summary>
        public void CalculateCheckpointGradients()
        {
            List<int> checkpoint_numbers = new List<int>(checkpoints_in.Keys);
            checkpoint_numbers.Sort();
            checkpoint_numbers.Reverse();
            foreach (int nr in checkpoint_numbers)
            {
                checkpoints_in[nr].TransferDerivatives(checkpoints_out[nr]);
            }
        }


    }
}
