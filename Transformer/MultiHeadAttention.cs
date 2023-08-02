using System;
using Transformer.Utils;

namespace Transformer
{
    public class MultiHeadAttention
    {
        private bool mask;
        private int embeddingSize;
        private int nr_heads;       // Number of attention heads
        private int dk;             // Key dimension
        private int dv;             // Value dimension

        // Learned linear input layers
        private Tensor[] Qm, Km, Vm;

        // Learned output layer
        private Tensor Wo;

        private Optimizer[] QmOptimizer, KmOptimizer, VmOptimizer;
        private Optimizer WoOptimizer;


        public MultiHeadAttention(int dk, int dv, int nr_heads, int embeddingSize, bool mask)
        {
            this.dk = dk;
            this.dv = dv;
            this.nr_heads = nr_heads;
            this.embeddingSize = embeddingSize;
            this.mask = mask;

            InitializeLinearFilters();
            InitalizeOptimizers();
        }

        public Tensor Update(Tensor inputData)
        {
            ApplyLinearInputFilters(inputData, out Tensor[] Qf, out Tensor[] Kf, out Tensor[] Vf);
            return CalculateScaledMultiHeadedAttention(Qf, Kf, Vf);
        }

        public Tensor Update(Tensor encoderOutput, Tensor queries)
        {
            ApplyLinearInputFilters(encoderOutput, queries, out Tensor[] Qf, out Tensor[] Kf, out Tensor[] Vf);
            return CalculateScaledMultiHeadedAttention(Qf, Kf, Vf);
        }

        private void ApplyLinearInputFilters(Tensor inputData, out Tensor[] Qf, out Tensor[] Kf, out Tensor[] Vf)
        {
            Qf = new Tensor[nr_heads];
            Kf = new Tensor[nr_heads];
            Vf = new Tensor[nr_heads];

            for (int h = 0; h < nr_heads; h++)
            {
                Qf[h] = Tensor.MatMul(inputData, Qm[h]);
                Kf[h] = Tensor.MatMul(inputData, Km[h]);
                Vf[h] = Tensor.MatMul(inputData, Vm[h]);
            }
        }

        private void ApplyLinearInputFilters(Tensor encoderOutput, Tensor queries, out Tensor[] Qf, out Tensor[] Kf, out Tensor[] Vf)
        {
            Qf = new Tensor[nr_heads];
            Kf = new Tensor[nr_heads];
            Vf = new Tensor[nr_heads];

            for (int h = 0; h < nr_heads; h++)
            {
                Qf[h] = Tensor.MatMul(queries, Qm[h]);
                Kf[h] = Tensor.MatMul(encoderOutput, Km[h]);
                Vf[h] = Tensor.MatMul(encoderOutput, Vm[h]);
            }
        }

        private Tensor CalculateScaledMultiHeadedAttention(Tensor[] Qf, Tensor[] Kf, Tensor[] Vf)
        {
            Tensor[] AttentionHeads = new Tensor[nr_heads];

            for (int h = 0; h < nr_heads; h++)
            {
                Tensor AttentionFilter = Tensor.MatMul(Qf[h], Kf[h].Transpose());
                Tensor scaledAttentionFilter = AttentionFilter.Scale(1 / Math.Sqrt(dk));
                if (mask)
                    scaledAttentionFilter.Mask();
                AttentionHeads[h] = Tensor.MatMul(scaledAttentionFilter.Softmax(), Vf[h]);
            }

            // Apply linear output layer to get the correct output size
            Tensor C = Tensor.Concat(AttentionHeads);
            return Tensor.MatMul(C, Wo);

        }

        public void MakeTrainingStep(double learningRate, int step)
        {
            for (int h = 0; h < nr_heads; h++)
            {
                KmOptimizer[h].MakeTrainingStep(learningRate, step, Km[h]);
                QmOptimizer[h].MakeTrainingStep(learningRate, step, Qm[h]);
                VmOptimizer[h].MakeTrainingStep(learningRate, step, Vm[h]);
            }
            WoOptimizer.MakeTrainingStep(learningRate, step, Wo);
        }

        private void InitializeLinearFilters()
        {
            Qm = new Tensor[nr_heads];
            Km = new Tensor[nr_heads];
            Vm = new Tensor[nr_heads];

            for (int h = 0; h < nr_heads; h++)
            {
                Qm[h] = new Tensor(embeddingSize, dk);
                Km[h] = new Tensor(embeddingSize, dk);
                Vm[h] = new Tensor(embeddingSize, dv);

                Qm[h].GenerateNormalRandomValues();
                Km[h].GenerateNormalRandomValues();
                Vm[h].GenerateNormalRandomValues();
            }

            Wo = new Tensor(dv * nr_heads, embeddingSize);
            Wo.GenerateNormalRandomValues();
        }

        private void InitalizeOptimizers()
        {
            QmOptimizer = new Optimizer[nr_heads];
            KmOptimizer = new Optimizer[nr_heads];
            VmOptimizer = new Optimizer[nr_heads];

            for (int h = 0; h < nr_heads; h++)
            {
                QmOptimizer[h] = new Optimizer(Qm[h]);
                KmOptimizer[h] = new Optimizer(Km[h]);
                VmOptimizer[h] = new Optimizer(Vm[h]);
            }

            WoOptimizer = new Optimizer(Wo);
        }

    }
}
