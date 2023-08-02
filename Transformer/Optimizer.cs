using System;
using Transformer.Utils;

namespace Transformer
{
    /// <summary>
    /// Implements the Adam optimizer
    /// </summary>
    public class Optimizer
    {
        private const double beta1 = 0.9;
        private const double beta2 = 0.999;
        private const double eps = 1e-8;
        
        private Tensor M;
        private Tensor V;

        public Optimizer(Tensor T)
        {
            M = new Tensor(T) * 0;
            V = new Tensor(T) * 0;
        }

        public void MakeTrainingStep(double learningRate, int step, Tensor T)
        {
            M = beta1 * M + (1.0 - beta1) * T.GetDerivatives();
            V = beta2 * V + (1.0 - beta2) * T.GetDerivatives().Pow(2);
            var m_hat = M / (1.0 - Math.Pow(beta1, step));
            var v_hat = V / (1.0 - Math.Pow(beta2, step));
            var correction = -learningRate * m_hat / (v_hat.Pow(0.5) + eps);
            T.MatAdd(correction);
            T.ClearDerivatives();
        }


    }
}
